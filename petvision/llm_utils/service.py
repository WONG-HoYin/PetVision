import json
import re
import requests
from typing import Dict, Optional, Any

import os
api_key = os.getenv("MISTRAL_API_KEY", "")
MODEL_ID = "pixtral-12b-2409"
API_URL = "https://api.mistral.ai/v1/chat/completions"


def robust_json_cleaner(content: str) -> Optional[Dict]:
    """強力清理並解析 LLM 回傳的 JSON"""
    if not content or not isinstance(content, str):
        return None
    
    try:
        json_match = re.search(r'(\{[\s\S]*\})', content)
        if not json_match:
            return None
        
        clean_json = json_match.group(1)
        clean_json = re.sub(r',\s*}', '}', clean_json)
        clean_json = re.sub(r',\s*]', ']', clean_json)
        clean_json = re.sub(r'[\n\r]+', ' ', clean_json)
        clean_json = re.sub(r'```json\s*|\s*```', '', clean_json)
        
        return json.loads(clean_json)
    except Exception as e:
        print(f"JSON Cleaner Error: {e}")
        return None


def fix_structured_analysis(raw_data: Dict) -> Dict:
    """確保 structured_analysis 是簡單字串列表"""
    if not isinstance(raw_data, dict):
        return raw_data

    for key in ["pet_analysis_content", "non_pet_analysis_content"]:
        content = raw_data.get(key)
        if isinstance(content, dict):
            sa = content.get("structured_analysis")
            if isinstance(sa, list):
                new_sa = []
                for item in sa:
                    if isinstance(item, str):
                        new_sa.append(item)
                    elif isinstance(item, dict):
                        new_sa.append(str(item.get("description", item)))
                    else:
                        new_sa.append(str(item))
                content["structured_analysis"] = new_sa
    return raw_data


def get_llm_structured_analysis(
    cnn_results: Dict, 
    image_b64: str = None, 
    language: str = "ZH",
    confidence: float = 70.0,
    species_match: bool = True
) -> Optional[Dict[str, Any]]:
    
    try:
        from llm_utils.prompts import create_humanized_prompt
        
        messages = create_humanized_prompt(
            cnn_results, 
            confidence=confidence, 
            species_match=species_match, 
            language=language 
        )

        prompt_text = messages[-1]['content']

        lang_name = "Traditional Chinese (繁體中文)" if language == "ZH" else "English"
        force_lang = f"\n\nIMPORTANT: ALL text in the JSON must be in {lang_name} only. Do not mix languages."

        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text + force_lang},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.25,
            "max_tokens": 1200
        }

        response = requests.post(
            API_URL, 
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"}, 
            json=payload, 
            timeout=90
        )
        
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return None

        raw_content = response.json()['choices'][0]['message']['content']
        
        print("=== RAW LLM RESPONSE ===")
        print(raw_content[:2500])

        raw_data = robust_json_cleaner(raw_content)
        if not raw_data:
            return {"case_type": "special_case", "is_pet": False, "error": "JSON parse failed"}

        raw_data = fix_structured_analysis(raw_data)

        case_type = str(raw_data.get("case_type", "")).lower()
        raw_data["case_type"] = "normal_case" if "normal" in case_type else "special_case"

        return raw_data

    except Exception as e:
        print(f" LLM Error: {e}")
        return None


def format_analysis_for_display(res: Dict, language: str = "ZH") -> str:
    """顯示報告"""
    if not res or not isinstance(res, dict):
        return "無法生成報告，請再試一次。" if language == "ZH" else "Failed to generate report. Please try again."

    is_zh = (language == "ZH")
    
    pet_content = res.get("pet_analysis_content")
    non_pet = res.get("non_pet_analysis_content")

    if non_pet and isinstance(non_pet, dict):
        title = "### 特殊案例分析報告" if is_zh else "### Special Case Report"
        intro = non_pet.get("non_pet_intro_line", " 特殊案例")
        desc = non_pet.get("image_description_line", "")
        items = non_pet.get("structured_analysis", [])
        structured_text = "\n\n".join([f"• {item}" for item in items if isinstance(item, str)])
        return f"{title}\n\n**{intro}**\n\n{desc}\n\n{structured_text}".strip()

    if pet_content and isinstance(pet_content, dict):
        title = "### AI 寵物專家深度報告" if is_zh else "### AI Pet Expert Deep Report"
        intro = pet_content.get("intro_species_line", "")
        breed_line = pet_content.get("breed_line", "")
        items = pet_content.get("structured_analysis", [])
        structured_text = "\n\n".join([f"• {item}" for item in items if isinstance(item, str)])
        
        output = f"{title}\n\n**{intro}**\n\n{breed_line}\n\n{structured_text}".strip()
        return output

    return "分析完成，但格式異常。" if is_zh else "Analysis completed with minor format issues."

def get_llm_chat_response(query: str, cnn_results: Dict, chat_history: list, image_b64: str, language: str, initial_report: str) -> str:
    """ 
    處理追問對話：強迫 AI 扮演報告的作者，並以報告內容為唯一事實來源。
    """
    try:
        is_zh = (language == "ZH")
        
        system_msg = (
            f"你是一位專業的寵物專家。你剛剛為使用者寫了一份關於這張圖片的『深度分析報告』。\n"
            f"現在使用者要針對你寫的內容進行追問。你的回答必須符合以下原則：\n"
            f"1. **完全一致性**：你的回答必須與你剛才寫的報告事實完全一致，不可產生矛盾。\n"
            f"2. **上下文連結**：如果使用者提到『報告中說...』或詢問細節（如 CNN 信心度），請直接從下方的報告原文中提取答案。\n"
            f"3. **專業解讀**：如果使用者問為什麼，請結合圖片細節與報告中的邏輯來解釋。\n\n"
            f"【你剛才生成的報告原文如下】：\n"
            f"--- START OF REPORT ---\n"
            f"{initial_report}\n"
            f"--- END OF REPORT ---\n\n"
            f"請使用 {'繁體中文' if is_zh else 'English'} 進行對話。"
        )

        messages = [{"role": "system", "content": system_msg}]
        
        for msg in chat_history[-5:]:
            messages.append(msg)

        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        response = requests.post(API_URL, headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"}, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return "對不起，我正在重新閱讀剛才的報告，請稍等。"
        
    except Exception as e:
        return f"對話系統暫時出現問題: {e}"