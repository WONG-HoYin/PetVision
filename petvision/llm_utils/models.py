from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class HumanReadablePetAnalysis(BaseModel):
    intro_species_line: str = Field(description="用一句溫暖的話描述看到的動物")
    breed_line: str = Field(description="對比 CNN 預測與實際觀察")
    structured_analysis: List[str] = Field(
        default_factory=list, 
        description="固定格式的分析列表"
    )
    
    # Optional fields for future expansion
    age_gender_summary_line: Optional[str] = None
    personality_line: Optional[str] = None
    emotion_line: Optional[str] = None
    health_line: Optional[str] = None


class HumanReadableNonPetAnalysis(BaseModel):
    non_pet_intro_line: str
    image_description_line: str
    invitation_line: str
    structured_analysis: List[str] = Field(default_factory=list)


class StructuredHumanizedOutput(BaseModel):
    case_type: Literal["normal_case", "special_case"]
    is_pet: bool
    
    pet_analysis_content: Optional[HumanReadablePetAnalysis] = None
    non_pet_analysis_content: Optional[HumanReadableNonPetAnalysis] = None