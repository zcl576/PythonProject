from pydantic import BaseModel


class UserContext(BaseModel):
    project_id: int