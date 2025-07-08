from fastapi import FastAPI
from pydantic import BaseModel
import torch
from fire_model import FireSpreadModel


app = FastAPI()

# Load model
model = FireSpreadModel()
model.eval()

# Request format
class InputData(BaseModel):
    raster: list    # flattened raster [channels*H*W]
    timeseries: list # flattened timeseries [T*2]

@app.post("/predict")
def predict(data: InputData):
    # reshape raster
    raster_tensor = torch.tensor(data.raster, dtype=torch.float32).view(1, 3, 64, 64)
    
    # reshape timeseries
    timeseries_tensor = torch.tensor(data.timeseries, dtype=torch.float32).view(1, -1, 2)
    
    fire_prob, fire_mask = model(raster_tensor, timeseries_tensor)
    
    # convert to lists
    mask_list = fire_mask.squeeze().detach().numpy().tolist()
    
    return {
        "fire_probability": fire_prob.item(),
        "fire_mask": mask_list
    }
