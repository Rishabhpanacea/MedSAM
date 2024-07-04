import os.path
import tempfile

from fastapi import FastAPI, HTTPException,APIRouter, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel, Field

from utils.Ploting_Utils import show_box, show_mask
from src.models.transformer import MEDSAM

import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch


router = APIRouter()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Update with your frontend URL
]

class SegmentRequestData(BaseModel):
    Seg: list = Field(..., example=[[1.0, 2.0], [3.0, 4.0]])
    SOP: str = Field(..., example="1.3.12.2.1107.5.1.4.73214.30000021013004473469000030614")

class SegmentRequest(BaseModel):
    data: SegmentRequestData

class RequestData(BaseModel):
    pixelData: list = Field(..., example=[1.0, 2.0, 3.0])
    segmentRequest: SegmentRequest


async def process_pixel_data(pixel_data: list):
    if len(pixel_data) != 262144:
        raise HTTPException(status_code=400, detail='Incorrect data size. Expected 512x512 pixels.')

    pixel_array = np.array(pixel_data, dtype=np.uint8).reshape((512, 512))

    # imageio.imwrite('pixel_data.png', pixel_array)
    
    # Plot the pixel_array using matplotlib (optional)
    plt.imshow(pixel_array, cmap='gray')  # Choose a colormap (e.g., 'gray' for grayscale)
    plt.title('Received Pixel Data')
    plt.colorbar()
    plt.show()
    
    return {'message': 'Pixel data processed successfully',"pixel_data":pixel_array}

# Function to perform segmentation
async def perform_segmentation(segment_request: SegmentRequest, pixel_data_output):
    try:
        model = MEDSAM()
        data = segment_request.data
        seg_data = data.Seg
        sop = data.SOP

        # img_path = rf"pixel_data.png"
        # img_np = plt.imread(img_path)
        img_np = pixel_data_output["pixel_data"]

        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        H, W, _ = img_3c.shape

        box_np = np.array([[seg_data[0][1], seg_data[0][0], seg_data[1][1], seg_data[1][0]]])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        # Perform the segmentation inference
        medsam_seg = model.medsam_inference(img_np, box_1024)
        medsam_seg = medsam_seg.cpu().numpy() if isinstance(medsam_seg, torch.Tensor) else medsam_seg

        # Flatten the segmentation mask for the response
        flattened_segmentation = medsam_seg.flatten().tolist()

        # Plot and show the results
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        show_box(box_np[0], ax[0])
        ax[0].set_title("Input Image and Bounding Box")
        ax[1].imshow(img_3c)
        show_mask(medsam_seg, ax[1])
        show_box(box_np[0], ax[1])
        ax[1].set_title("MedSAM Segmentation")
        plt.show()

        # Return the segmentation result
        return {"segmentation_result": flattened_segmentation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Combined endpoint to handle both pixel data and segmentation requests
@router.post("/predict/")
async def process_data(request_data: RequestData):
    try:
        # Process pixel data
        pixel_data_output = await process_pixel_data(request_data.pixelData)

        # Perform segmentation
        segmentation_result = await perform_segmentation(request_data.segmentRequest, pixel_data_output)
        
        return segmentation_result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





























































# class PredictionRequest(BaseModel):
#     Seg: List[List[float]]
#     SOP: str


# @router.post("/predict/")
# async def create_prediction(payload: PredictionRequest):
#     model = SegFormer()
#     fd, dicom_path = tempfile.mkstemp(suffix=".dcm", dir=DICOM_TEMP_PATH, prefix='tmp')
#     try:
#         with os.fdopen(fd, 'wb') as tmp:
#             data = await file.read()
#             tmp.write(data)
#         jpeg_path = saveDicomAsJPEG(dicom_path)
#         sop_instance_uid, study_instance_uid, series_instance_uid \
#             = getDicomFIleIds(dicom_path)
#         seg_info = model.classify(jpeg_path)
#         response = ResponseUtils.getOHIFObjects(sop_instance_uid, study_instance_uid, series_instance_uid, seg_info)
#         return response
#     finally:
#         # Whether we had an error or not, it's important to clean up the temp file
#         os.remove(dicom_path)
#         os.remove(jpeg_path)