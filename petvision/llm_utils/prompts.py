from typing import List, Dict


BREED_LIST = """
Frog, Salamander, Toad, Abyssinian, American Bobtail, American Curl, American Shorthair, 
American Wirehair, Applehead Siamese, Balinese, Bengal, Birman, Bombay, British Shorthair, 
Burmese, Burmilla, Calico, Canadian Hairless, Chartreux, Chausie, Chinchilla, Cornish Rex, 
Cymric, Devon Rex, Dilute Calico, Dilute Tortoiseshell, Domestic Long Hair, Domestic Medium Hair, 
Domestic Short Hair, Egyptian Mau, Exotic Shorthair, Extra Toes Cat Hemingway Polydactyl, 
Havana, Himalayan, Japanese Bobtail, Javanese, Korat, LaPerm, Maine Coon, Manx, Munchkin, 
Nebelung, Norwegian Forest Cat, Ocicat, Oriental Long Hair, Oriental Short Hair, Oriental Tabby, 
Persian, Pixiebob, Ragamuffin, Ragdoll, Russian Blue, Scottish Fold, Selkirk Rex, Siamese, 
Siberian, Silver, Singapura, Snowshoe, Somali, Sphynx, Tabby, Tiger, Tonkinese, Torbie, 
Tortoiseshell, Turkish Angora, Turkish Van, Tuxedo, York Chocolate, Affenpinscher, Afghan Hound, 
African Hunting Dog, Airedale, American Bulldog, American Pit Bull Terrier, American Staffordshire Terrier, 
Appenzeller, Australian Terrier, Basenji, Basset, Beagle, Bedlington Terrier, Bernese Mountain Dog, 
And Tan Coonhound, Blenheim Spaniel, Bloodhound, Bluetick, Border Collie, Border Terrier, Borzoi, 
Boston Bull, Bouvier Des Flandres, Boxer, Brabancon Griffon, Briard, Brittany Spaniel, Bull Mastiff, 
Cairn, Cardigan, Chesapeake Bay Retriever, Chihuahua, Chow, Clumber, Cocker Spaniel, Collie, 
Coated Retriever, Dandie Dinmont, Dhole, Dingo, Doberman, English Cocker Spaniel, English Foxhound, 
English Setter, English Springer, Entlebucher, Eskimo Dog, Flat-Coated Retriever, French Bulldog, 
German Shepherd, Haired Pointer, Giant Schnauzer, Golden Retriever, Gordon Setter, Great Dane, 
Great Pyrenees, Greater Swiss Mountain Dog, Groenendael, Ibizan Hound, Irish Setter, Irish Terrier, 
Irish Water Spaniel, Irish Wolfhound, Italian Greyhound, Japanese Spaniel, Keeshond, Kelpie, 
Kerry Blue Terrier, Komondor, Kuvasz, Labrador Retriever, Lakeland Terrier, Leonberg, Lhasa, 
Malamute, Malinois, Maltese Dog, Mexican Hairless, Miniature Pinscher, Miniature Poodle, 
Miniature Schnauzer, Newfoundland, Norfolk Terrier, Norwegian Elkhound, Norwich Terrier, 
Old English Sheepdog, Otterhound, Papillon, Pekinese, Pembroke, Pomeranian, Pug, Redbone, 
Rhodesian Ridgeback, Rottweiler, Saint Bernard, Saluki, Samoyed, Schipperke, Scottish Deerhound, 
Scotch Terrier, Sealyham Terrier, Shetland Sheepdog, Shiba Inu, Tzu, Siberian Husky, Silky Terrier, 
Coated Wheaten Terrier, Staffordshire Bullterrier, Standard Poodle, Standard Schnauzer, Sussex Spaniel, 
Tibetan Mastiff, Tibetan Terrier, Toy Poodle, Toy Terrier, Vizsla, Walker Hound, Weimaraner, 
Welsh Springer Spaniel, West Highland White Terrier, Whippet, Haired Fox Terrier, Yorkshire Terrier, 
Chameleon, Gecko, Iguana, Lizard, Snake, TurtleTortoise, Antelope, Badger, Bat, Bear, Bee, Beetle, 
Bison, Boar, Butterfly, Caterpillar, Chimpanzee, Cockroach, Cow, Coyote, Crab, Crow, Deer, Dolphin, 
Donkey, Dragonfly, Duck, Eagle, Elephant, Flamingo, Fly, Fox, Goat, Goldfish, Goose, Gorilla, 
Grasshopper, Hamster, Hare, Hedgehog, Hippopotamus, Hornbill, Horse, Hummingbird, Hyena, Jellyfish, 
Kangaroo, Koala, Ladybugs, Leopard, Lion, Lobster, Mosquito, Moth, Mouse, Octopus, Okapi, Orangutan, 
Otter, Owl, Ox, Oyster, Panda, Parrot, Pelecaniformes, Penguin, Pig, Pigeon, Porcupine, Possum, 
Raccoon, Rat, Reindeer, Rhinoceros, Sandpiper, Seahorse, Seal, Shark, Sheep, Sparrow, Squid, 
Squirrel, Starfish, Swan, Turkey, Turtle, Whale, Wolf, Wombat, Woodpecker, Zebra, Other, 
Food, Toy, Statue, Roasted Pig, Ceramic Figurine, Painting, Plush Toy
"""

PROBLEMATIC_BREEDS = {
    "American Bobtail", "American Curl", "American Shorthair", "American Wirehair",
    "Balinese", "Birman", "Burmese", "Burmilla", "Canadian Hairless",
    "Chartreux", "Chausie", "Chinchilla", "Cornish Rex", "Cymric",
    "Devon Rex", "Dilute Calico", "Dilute Tortoiseshell", "Domestic Long Hair",
    "Domestic Medium Hair", "Domestic Short Hair", "Extra Toes Cat Hemingway Polydactyl",
    "LaPerm", "Manx", "Munchkin", "Nebelung", "Oriental Long Hair",
    "Oriental Short Hair", "Pixiebob", "Selkirk Rex",
    "Australian Terrier", "Bedlington Terrier", "Borzoi", "Boston Bull",
    "Bouvier Des Flandres", "Brabancon Griffon", "Brittany Spaniel", "Cairn Terrier",
    "Clumber Spaniel", "English Setter", "English Springer Spaniel", "Flat-Coated Retriever"
}

def create_humanized_prompt(
    cnn_results: Dict, 
    confidence: float = 70.0, 
    species_match: bool = True, 
    language: str = "ZH"
) -> List[Dict]:
    
    lang_instruction = "Please respond in Traditional Chinese (繁體中文)." if language == "ZH" else "Please respond in English."
    
    breed = cnn_results.get('cnn_guess', 'Unknown')
    breed_conf = cnn_results.get('breed_conf', confidence)

    is_problematic = breed in PROBLEMATIC_BREEDS

    INDEPENDENT_THINKING_CMD = (
        "核心原則：你必須具備獨立判斷能力，不要盲目跟隨 CNN 的預測結果。"
        "如果圖片中明顯不是生物（例如是食物、玩具、藝術品），或者品種明顯錯誤，你必須果斷糾正並給予合理的視覺證據。"
        if language == "ZH" else 
        "Core Principle: You must exercise independent judgment. DO NOT blindly follow the CNN prediction. "
        "If the image clearly shows a non-living object (e.g., food, toy, artwork) or an incorrect breed, "
        "you must decisively correct it with visual evidence."
    )


    PROBLEMATIC_WARNING = f"""
【重要警示 - CNN 高誤判品種】此預測品種「{breed}」屬於 CNN 模型最常誤判的清單。
請**極度嚴格**依據圖片中的明確視覺特徵進行獨立判斷：
• 耳型、尾巴形狀、臉型、毛質/毛長、體型比例、四肢長度、眼睛形狀等關鍵特徵
• 除非圖片清楚顯示該品種的典型 signature features，否則請考慮調整為更常見的品種（如 Domestic Short Hair、Tabby、Mixed breed 等）
""" if is_problematic else ""

    json_format_instruction = """
請**嚴格**按照以下 JSON 格式回應，不要加入任何額外文字、Markdown 或解釋。
structured_analysis 必須是簡單的字串陣列（List[str]），不要使用物件。
所有文字內容必須完全使用指定的語言（繁體中文或 English），不可混用。
"""

    if not species_match:
        system_prompt = f"你是一位嚴謹的動物學家。當 CNN 預測與物種不一致時必須強力修正。{lang_instruction}"
        user_prompt = f"""這是邏輯衝突案例。CNN 預測：{breed} ({breed_conf:.1f}%)，但物種明顯不同。

{INDEPENDENT_THINKING_CMD}
{PROBLEMATIC_WARNING}

{json_format_instruction}
請輸出以下格式的 JSON：
{{
  "case_type": "special_case",
  "identified_breed": "從名單中選出最符合圖片的真實內容",
  "is_pet": false,
  "pet_analysis_content": null,
  "non_pet_analysis_content": {{
    "non_pet_intro_line": " 嚴重邏輯衝突 - 已修正",
    "image_description_line": "圖片中實際是[真實內容]，CNN 錯誤預測為 {breed}。",
    "invitation_line": "請上傳清晰的家養寵物照片。",
    "structured_analysis": ["項目1", "項目2"]
  }}
}}"""

    elif breed_conf < 60:
        system_prompt = f"你是一位細心的動物辨識助手與獸醫知識豐富的分析師。{lang_instruction}"
        user_prompt = f"""這是低信心真實案例。CNN 預測：{breed} ({breed_conf:.1f}%)。

{INDEPENDENT_THINKING_CMD}
{PROBLEMATIC_WARNING}

**重要分析步驟（請嚴格遵守）**：
1. 先簡述該品種典型的健康特徵和常見健康問題。
2. 再根據圖片判斷這隻個體的健康狀況。
3. 最後給出實用建議。

{json_format_instruction}
請輸出以下格式的 JSON：
{{
  "case_type": "normal_case",
  "identified_breed": "從名單中選出最符合的品種",
  "is_pet": true,
  "pet_analysis_content": {{
    "intro_species_line": "這是一隻[品種]，雖然 CNN 信心較低，但視覺證據支持。",
    "breed_line": "模型預測 {breed}，經檢查基本正確或輕微調整。",
    "structured_analysis": [
      "確認為真實活體動物",
      "品種鑑定：{breed}",
      "低信心原因：光線、角度或品種特徵不明顯",
      "體型：請依據實際圖片描述",
      "毛髮：請依據實際圖片描述",
      "年齡估計：請依據實際圖片判斷",
      "健康觀察：狀態良好，無明顯異常",
      "品種典型健康特徵：{breed} 常見健康情況為 [簡述該品種的典型健康特徵及常見問題]",
      "這隻個體健康判斷：根據圖片判斷，這隻{breed}的健康狀況為 [健康 / 有輕微關注點 / 需注意某問題]",
      "照護建議：針對此品種，建議定期檢查 [具體建議]"
    ]
  }},
  "non_pet_analysis_content": null
}}"""

    else:
        system_prompt = f"你是一位溫暖專業的寵物專家與獸醫知識豐富的分析師。{lang_instruction}"
        user_prompt = f"""這是正常穩定識別案例。CNN 預測：{breed} ({breed_conf:.1f}%)，物種品種一致。

{INDEPENDENT_THINKING_CMD}
{PROBLEMATIC_WARNING}

**重要分析步驟（請嚴格遵守）**：
1. 先回想並簡述該品種典型的健康特徵和常見健康問題。
2. 再根據圖片中動物的外觀，判斷這隻個體的健康狀況。
3. 最後給出實用建議。

{json_format_instruction}
請嚴格輸出以下格式的 JSON：
{{
  "case_type": "normal_case",
  "identified_breed": "{breed}",
  "is_pet": true,
  "pet_analysis_content": {{
    "intro_species_line": "這是一隻可愛的[品種]！",
    "breed_line": "CNN 以 {breed_conf:.1f}% 信心正確識別為 {breed}。",
    "structured_analysis": [
      "確認為真實活體動物",
      "品種鑑定：{breed}",
      "體型：請依據實際圖片描述",
      "毛髮：請依據實際圖片描述",
      "年齡估計：請依據實際圖片判斷",
      "健康觀察：狀態良好，無明顯異常",
      "品種典型健康特徵：{breed} 常見健康情況為 [簡述該品種的典型健康特徵及常見問題]",
      "這隻個體健康判斷：根據圖片判斷，這隻{breed}的健康狀況為 [健康 / 有輕微關注點 / 需注意某問題]",
      "照護建議：針對此品種，建議定期檢查 [具體建議]"
    ]
  }},
  "non_pet_analysis_content": null
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages