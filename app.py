from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import random

# Assuming these are imported functions from your provided project code
from mini_project_code import recolduser, recnewuser

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load dataset for user information
#dataset_path = "marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv"
train_data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')

# User data structure in-memory (this can be stored in the dataset in production)
users = {}

# Generate a new user ID
def generate_user_id():
    return random.randint(1000, 9999)

# Save new user to the dataset
def save_new_user(user_id, tags):
    global train_data
    new_user_entry = {
        'Uniq Id': user_id,
        'Product Tags': tags
    }
    train_data = pd.concat([train_data, pd.DataFrame([new_user_entry])], ignore_index=True)

@app.get("/", response_class=JSONResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register", response_class=JSONResponse)
async def register_new_user(request: Request):
    data = await request.json()
    tags = data.get("tags")
    user_id = generate_user_id()
    save_new_user(user_id, tags)
    
    # Run recommendation for new user based on tags
    recommended_products = recnewuser(tags)
    if recommended_products is None:
        return {"user_id": user_id, "products": []}  # Return an empty list if no products are found
    return {"user_id": user_id, "products": recommended_products.to_dict(orient='records')}

@app.post("/login", response_class=JSONResponse)
async def login_old_user(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    if user_id in train_data['Uniq Id'].values:
        return {"status": "success"}
    return {"status": "error", "message": "User ID not found"}

@app.post("/search", response_class=JSONResponse)
async def search_products(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    item_index = train_data[train_data['Uniq Id'] == user_id].index[0]
    description = data.get("description")
    # Run recommendation function based on description
    recommended_products = recolduser(item_index, description)
    
    return {"products": recommended_products.to_dict(orient='records')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
