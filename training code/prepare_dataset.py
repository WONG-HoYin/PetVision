import os
import cv2
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ================= Configuration =================
base_dir = os.path.dirname(os.path.abspath(__file__))


CONTAINER_BASE_DIR = "/workspace"


oxford_dir = os.path.join(base_dir, "the-oxfordiiit-pet-dataset", "images")

kaggle_dir = os.path.join(base_dir, "kaggle-dog-dataset", "images") 

cat_dir = os.path.join(base_dir, "cat-breeds-dataset", "images")

reptile_dir = os.path.join(base_dir, "reptiles-amphibians-dataset")

animals_dir = os.path.join(base_dir, "animals-90-dataset")

csv_path = os.path.join(base_dir, "attributes.csv")
output_csv_path = os.path.join(base_dir, "final_dataset.csv")


MAX_IMAGES_PER_BREED = 200
MIN_IMAGES_PER_BREED = 50


class DatasetStats:
    def __init__(self):
        self.total_scanned = defaultdict(int)    
        self.filtered_extension = defaultdict(int)
        self.filtered_csv_match = defaultdict(int)
        self.filtered_max_limit = defaultdict(int) 
        self.collected = defaultdict(int)        
        self.missing_breeds = set()     
        
    def print_summary(self):
        print("\n" + "="*70)
        print(" 数据筛选详细统计报告")
        print("="*70)
        
        total_all = sum(self.total_scanned.values())
        collected_all = sum(self.collected.values())
        
        print(f"\n【总体情况】")
        print(f"  扫描文件总数: {total_all:,}")
        print(f"  最终收集数量: {collected_all:,}")
        print(f"  总体保留率: {collected_all/total_all*100:.1f}%")
        
        print(f"\n【各数据源详情】")
        for source in self.total_scanned.keys():
            print(f"\n  {source}:")
            print(f"    扫描文件: {self.total_scanned[source]:,}")
            print(f"    扩展名过滤: {self.filtered_extension[source]:,} ({self.filtered_extension[source]/max(self.total_scanned[source],1)*100:.1f}%)")
            print(f"    CSV不匹配: {self.filtered_csv_match[source]:,}")
            print(f"    超过上限过滤: {self.filtered_max_limit[source]:,}")
            print(f"    最终收集: {self.collected[source]:,}")
        
        if self.missing_breeds:
            print(f"\n【CSV匹配问题】")
            print(f"  共 {len(self.missing_breeds)} 个品种在 attributes.csv 中未找到:")
            for i, breed in enumerate(sorted(self.missing_breeds)[:20]):
                print(f"    {i+1}. {breed}")
            if len(self.missing_breeds) > 20:
                print(f"    ... 还有 {len(self.missing_breeds)-20} 个品种")


stats = DatasetStats()


def get_container_path(windows_path):
    r"""
    将Windows绝对路径转换为容器内路径
    格式: C:\Users\A\Desktop\PetVision_Data\... -> /workspace/...
    """

    path = str(windows_path)
    

    path = path.replace('\\', '/')
    

    windows_prefixes = [
        "C:/Users/A/Desktop/PetVision_Data/",
        "c:/Users/A/Desktop/PetVision_Data/",
        "C:/Users/A/Desktop/PetVision_Data\\",
        "c:/Users/A/Desktop/PetVision_Data\\"
    ]
    

    for prefix in windows_prefixes:
        clean_prefix = prefix.replace('\\', '/')
        if path.startswith(clean_prefix):
            relative_path = path[len(clean_prefix):]
            return f"/workspace/{relative_path}"
    

    return path

def apply_simple_augmentation(img):
    """Apply simple 2D augmentations without changing breed features."""
    h, w = img.shape[:2]
    

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    

    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random brightness adjustment
    brightness = random.uniform(0.9, 1.1)
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
    
    return img

if not os.path.exists(csv_path):
    print(f"Error: attributes.csv not found at {csv_path}!")
    exit()

df_attr = pd.read_csv(csv_path)
df_attr['Lookup_Key'] = df_attr['Breed'].str.lower().str.replace('_', ' ').str.replace('-', ' ').str.strip()
df_attr = df_attr.drop_duplicates(subset=['Lookup_Key'], keep='first')
attr_dict = df_attr.set_index('Lookup_Key').to_dict('index')

print(f"Loaded {len(attr_dict)} breeds from CSV.")
print(f"注意：本脚本将生成Docker容器内路径格式（/workspace/...）")
print(f"在Windows上可查看CSV内容，但无法直接打开图片文件")
print(f"在Docker容器内可直接用于训练\n")



max_breed_count = 0
breed_counts = defaultdict(int)


for source_path, source_name in [
    (oxford_dir, "oxford"),
    (kaggle_dir, "kaggle_dogs"),
    (cat_dir, "cat_dataset"),
    (reptile_dir, "reptile_dataset"),
    (animals_dir, "animals_90_dataset")
]:
    if not os.path.exists(source_path):
        continue
    if source_name == "oxford":
        for filename in os.listdir(source_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            raw_name = filename.rsplit('_', 1)[0]
            lookup_key = raw_name.lower().replace('_', ' ').replace('-', ' ').strip()
            if lookup_key in attr_dict:
                breed_counts[lookup_key] += 1
    else:
        for folder in os.listdir(source_path):
            folder_path = os.path.join(source_path, folder)
            if not os.path.isdir(folder_path):
                continue
            images = [img for img in os.listdir(folder_path) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            breed_counts[folder] += len(images)


max_breed_count = min(200, max(breed_counts.values()) if breed_counts else 100)
print(f"Global max breed count: {max_breed_count} (target for balancing)")

valid_data = []


if os.path.exists(oxford_dir):
    print("Scanning Oxford Source...")
    breed_images = defaultdict(list)
    breed_info = {}
    breed_counts_tracker = {} 
    
    for filename in os.listdir(oxford_dir):
        stats.total_scanned['oxford'] += 1
        

        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            stats.filtered_extension['oxford'] += 1
            continue
            
        raw_name = filename.rsplit('_', 1)[0]
        lookup_key = raw_name.lower().replace('_', ' ').replace('-', ' ').strip()
        

        if lookup_key not in attr_dict:
            stats.filtered_csv_match['oxford'] += 1
            stats.missing_breeds.add(raw_name)
            continue
        
        info = attr_dict[lookup_key]
        breed_name = info['Breed']
        
        breed_images[breed_name].append(filename)
        breed_info[breed_name] = info 

    for breed_name, images in breed_images.items():
        info = breed_info[breed_name]
        

        if len(images) < 100:

            temp_aug_dir = os.path.join(oxford_dir, f"{breed_name}_augmented")
            os.makedirs(temp_aug_dir, exist_ok=True)
            

            for img in images:
                img_path = os.path.join(oxford_dir, img)
                img_data = cv2.imread(img_path)
                if img_data is None:
                    continue
                

                augmented_img = apply_simple_augmentation(img)
                

                augmented_img_path = os.path.join(temp_aug_dir, f"{os.path.splitext(img)[0]}_aug.jpg")
                cv2.imwrite(augmented_img_path, augmented_img)
                

                images.append(os.path.basename(augmented_img_path))


        for img in images:

            if breed_counts_tracker.get(breed_name, 0) >= MAX_IMAGES_PER_BREED:
                stats.filtered_max_limit['oxford'] += 1
                continue
            windows_path = os.path.join(oxford_dir, img)
            container_path = get_container_path(windows_path)
            valid_data.append({ 
                'filepath': container_path, 
                'breed_label': breed_name, 
                'species_label': info['Species'], 
                'size_label': info['Size'], 
                'coat_label': info['Coat'], 
                'source': 'oxford' 
            }) 
            stats.collected['oxford'] += 1
            breed_counts_tracker[breed_name] = breed_counts_tracker.get(breed_name, 0) + 1
    
    print(f" → 收集 {stats.collected['oxford']:,} 张图片\n")


def scan_folder_source(source_path, source_name):
    if not os.path.exists(source_path):
        print(f"Warning: {source_name} directory not found.")
        return 0
    print(f"Scanning {source_name}...")
    breed_counts_tracker = {}
    
    if source_name == "oxford":
        breed_images = defaultdict(list)
        breed_info = {}
        for filename in os.listdir(source_path):
            stats.total_scanned[source_name] += 1

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                stats.filtered_extension[source_name] += 1
                continue
            raw_name = filename.rsplit('_', 1)[0]
            lookup_key = raw_name.lower().replace('_', ' ').replace('-', ' ').strip()

            if lookup_key not in attr_dict:
                stats.filtered_csv_match[source_name] += 1
                stats.missing_breeds.add(raw_name)
                continue
            info = attr_dict[lookup_key]
            breed_name = info['Breed']
            breed_images[breed_name].append(filename)
            breed_info[breed_name] = info
        
        for breed_name, images in breed_images.items():
            info = breed_info[breed_name]
            target_count = min(200, max_breed_count) 
            current_count = len(images)
            
   
            if current_count < target_count:
                num_to_augment = target_count - current_count
                temp_aug_dir = os.path.join(source_path, f"{breed_name}_augmented")
                os.makedirs(temp_aug_dir, exist_ok=True)
                for i in range(num_to_augment):
                    img = random.choice(images)
                    img_path = os.path.join(source_path, img)
                    img_data = cv2.imread(img_path)
                    if img_data is None:
                        continue
                    augmented_img = apply_simple_augmentation(img_data)
                    augmented_img_path = os.path.join(temp_aug_dir, f"{os.path.splitext(img)[0]}_aug_{i}.jpg")
                    cv2.imwrite(augmented_img_path, augmented_img)
                    windows_path = augmented_img_path
                    container_path = get_container_path(windows_path)
                    valid_data.append({
                        'filepath': container_path,
                        'breed_label': breed_name,
                        'species_label': info['Species'],
                        'size_label': info['Size'],
                        'coat_label': info['Coat'],
                        'source': source_name
                    })
                    stats.collected[source_name] += 1
                    breed_counts_tracker[breed_name] = breed_counts_tracker.get(breed_name, 0) + 1
            

            for img in images:
                if breed_counts_tracker.get(breed_name, 0) >= MAX_IMAGES_PER_BREED:
                    stats.filtered_max_limit[source_name] += 1
                    continue
                windows_path = os.path.join(source_path, img)
                container_path = get_container_path(windows_path)
                valid_data.append({
                    'filepath': container_path,
                    'breed_label': breed_name,
                    'species_label': info['Species'],
                    'size_label': info['Size'],
                    'coat_label': info['Coat'],
                    'source': source_name
                })
                stats.collected[source_name] += 1
                breed_counts_tracker[breed_name] = breed_counts_tracker.get(breed_name, 0) + 1
        
        return stats.collected[source_name]
    
    else: 
        for folder in os.listdir(source_path):
            folder_path = os.path.join(source_path, folder)
            if not os.path.isdir(folder_path):
                continue
  
            raw_name = folder.split('-', 1)[1] if '-' in folder else folder
            lookup_key = raw_name.lower().replace('_', ' ').replace('-', ' ').strip()

            if lookup_key not in attr_dict:
                stats.filtered_csv_match[source_name] += 1
                stats.missing_breeds.add(raw_name)
                print(f"  Missing in CSV: {folder}")
                continue
            info = attr_dict[lookup_key]
            images = [img for img in os.listdir(folder_path) 
                     if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            target_count = min(200, max_breed_count)
            current_count = len(images)
            
 
            if current_count < target_count:
                num_to_augment = target_count - current_count

                temp_aug_dir = os.path.join(folder_path, "augmented")
                os.makedirs(temp_aug_dir, exist_ok=True)

                for i in range(num_to_augment):
                    img = random.choice(images)
                    img_path = os.path.join(folder_path, img)
                    img_data = cv2.imread(img_path)
                    if img_data is None:
                        continue
                    augmented_img = apply_simple_augmentation(img_data)
                    augmented_img_path = os.path.join(temp_aug_dir, f"aug_{i}.jpg")
                    cv2.imwrite(augmented_img_path, augmented_img)

                    windows_path = augmented_img_path
                    container_path = get_container_path(windows_path)
                    valid_data.append({
                        'filepath': container_path,
                        'breed_label': info['Breed'],
                        'species_label': info['Species'],
                        'size_label': info['Size'],
                        'coat_label': info['Coat'],
                        'source': source_name
                    })
                    stats.collected[source_name] += 1
                    breed_counts_tracker[info['Breed']] = breed_counts_tracker.get(info['Breed'], 0) + 1
            

            for img in images:
                if breed_counts_tracker.get(info['Breed'], 0) >= MAX_IMAGES_PER_BREED:
                    stats.filtered_max_limit[source_name] += 1
                    continue
                windows_path = os.path.join(folder_path, img)
                container_path = get_container_path(windows_path)
                valid_data.append({
                    'filepath': container_path,
                    'breed_label': info['Breed'],
                    'species_label': info['Species'],
                    'size_label': info['Size'],
                    'coat_label': info['Coat'],
                    'source': source_name
                })
                stats.collected[source_name] += 1
                breed_counts_tracker[info['Breed']] = breed_counts_tracker.get(info['Breed'], 0) + 1
        
        return stats.collected[source_name]


collected_dogs = scan_folder_source(kaggle_dir, "kaggle_dogs")
print(f" → 收集 {collected_dogs:,} 张图片\n")

collected_cats = scan_folder_source(cat_dir, "cat_dataset")
print(f" → 收集 {collected_cats:,} 张图片\n")

collected_reptiles = scan_folder_source(reptile_dir, "reptile_dataset")
print(f" → 收集 {collected_reptiles:,} 张图片\n")

collected_animals = scan_folder_source(animals_dir, "animals_90_dataset")
print(f" → 收集 {collected_animals:,} 张图片\n")


df_data = pd.DataFrame(valid_data)

if len(df_data) == 0:
    print("Error: No images found! Check paths and attributes.csv")
    exit()


stats.print_summary()

print("\n" + "="*70)
print("📋 最终数据集处理")
print("="*70)

print(f"\n收集到的总图片数: {len(df_data):,}")
print(f"品种数量: {df_data['breed_label'].nunique()}")

print(f"\n=== 过滤样本数少于 {MIN_IMAGES_PER_BREED} 的品种 ===")
breed_counts = df_data['breed_label'].value_counts()
minority_breeds = breed_counts[breed_counts < MIN_IMAGES_PER_BREED].index.tolist()

if minority_breeds:
    print(f"发现 {len(minority_breeds)} 个品种样本数不足:")
    for breed in sorted(minority_breeds)[:10]:
        print(f"  - {breed}: {breed_counts[breed]}张")
    if len(minority_breeds) > 10:
        print(f"  ... 还有 {len(minority_breeds)-10} 个品种")
    
    df_filtered_min = df_data[~df_data['breed_label'].isin(minority_breeds)]
    print(f"\n过滤前: {len(df_data):,} 张图片, {len(breed_counts)} 个品种")
    print(f"过滤后: {len(df_filtered_min):,} 张图片, {len(df_filtered_min['breed_label'].unique())} 个品种")
    print(f"因样本不足被移除: {len(df_data) - len(df_filtered_min):,} 张图片")
else:
    df_filtered_min = df_data
    print(f" 所有品种样本数都超过 {MIN_IMAGES_PER_BREED}，无需过滤")


breed_counts = df_filtered_min['breed_label'].value_counts()
valid_breeds = breed_counts[breed_counts >= 2].index
df_filtered = df_filtered_min[df_filtered_min['breed_label'].isin(valid_breeds)]


train_df, val_df = train_test_split(
    df_filtered, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_filtered['breed_label']
)

train_df['split'] = 'train' 
val_df['split'] = 'val'
final_df = pd.concat([train_df, val_df])
final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"\n 成功！数据集映射已保存至: {output_csv_path}")
print(f"\n【最终结果】")
print(f"  训练集: {len(train_df):,} 张图片")
print(f"  验证集: {len(val_df):,} 张图片")
print(f"  总样本: {len(final_df):,} 张图片")
print(f"  品种数: {final_df['breed_label'].nunique()}")
print(f"Oxford path: {oxford_dir}")
print(f"Kaggle path: {kaggle_dir}")
print(f"Cat path: {cat_dir}")
print(f"Reptile path: {reptile_dir}")
print(f"Animals path: {animals_dir}")