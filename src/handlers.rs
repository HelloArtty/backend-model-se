use actix_multipart::Multipart;
use actix_web::{web, Error, HttpResponse, Result};
use futures_util::StreamExt;
use image::{GenericImageView, RgbaImage};
use ndarray::Array4;
use std::fs::File;
use std::io::Write;
use tempfile::Builder;
use tract_onnx::prelude::*;
use uuid::Uuid;

pub async fn predict(mut payload: Multipart) -> Result<HttpResponse, Error> {
    // สร้างไดเรกทอรีชั่วคราวสำหรับการอัปโหลดไฟล์
    let upload_dir = Builder::new()
        .prefix("temp_uploads")
        .tempdir()
        .map_err(|e| {
            eprintln!("Failed to create temporary directory: {}", e);
            actix_web::error::ErrorInternalServerError("Could not create temp directory")
        })?;

    // เก็บไฟล์ที่อัปโหลด
    let mut filepath = String::new();

    while let Some(item) = payload.next().await {
        let mut field = item?;
        let filename = format!("{}.jpg", Uuid::new_v4());
        filepath = upload_dir
            .path()
            .join(filename)
            .to_string_lossy()
            .into_owned();

        let filepath_for_closure = filepath.clone();

        let mut f = web::block(move || File::create(&filepath_for_closure))
            .await?
            .map_err(|e| {
                eprintln!("Failed to create file: {}", e);
                actix_web::error::ErrorInternalServerError("Could not save file")
            })?;

        while let Some(chunk) = field.next().await {
            let data = chunk?;
            f = web::block(move || f.write_all(&data).map(|_| f))
                .await?
                .map_err(|e| {
                    eprintln!("Failed to write data to file: {}", e);
                    actix_web::error::ErrorInternalServerError("Could not write to file")
                })?;
        }
    }

    let filepath_clone = filepath.clone();

    // โหลดและประมวลผลภาพ
    let img = match image::open(&filepath_clone) {
        Ok(img) => {
            let (width, height) = img.dimensions();
            // println!("Image dimensions: {}x{}", width, height);

            // Resize ภาพให้มีขนาดสูงสุดเป็น 224 ในอัตราส่วนที่ไม่ผิดเพี้ยน
            let (new_width, new_height) = if width > height {
                (224, (224 * height) / width)
            } else {
                ((224 * width) / height, 224)
            };

            let resized = img.resize(new_width, new_height, image::imageops::FilterType::Triangle);

            // สร้างภาพใหม่ที่มีขนาดเป็น 224x224 โดยการเติมขอบ (padding) ถ้าจำเป็น
            let mut final_image = RgbaImage::new(224, 224);
            let (resized_width, resized_height) = resized.dimensions();

            let pad_x = (224 - resized_width) / 2;
            let pad_y = (224 - resized_height) / 2;

            for y in 0..resized_height {
                for x in 0..resized_width {
                    let pixel = resized.get_pixel(x, y);
                    final_image.put_pixel(
                        x + pad_x,
                        y + pad_y,
                        image::Rgba([pixel[0], pixel[1], pixel[2], 255]),
                    );
                }
            }

            final_image
        }
        Err(e) => {
            eprintln!("Failed to process image: {}", e);
            return Err(actix_web::error::ErrorBadRequest("Invalid image file"));
        }
    };

    // แปลงภาพเป็นเทนเซอร์
    let mean = vec![0.485, 0.456, 0.406];
    let std = vec![0.229, 0.224, 0.225];

    let mut input_tensor = Array4::zeros((1, 3, 224, 224));
    for y in 0..224u32 {
        for x in 0..224u32 {
            let pixel = img.get_pixel(x, y);
            for c in 0..3 {
                let value = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                input_tensor[[0, c, y as usize, x as usize]] = value;
            }
        }
    }

    // โหลดและรันโมเดล ONNX
    let model = tract_onnx::prelude::onnx()
        .model_for_path("model.onnx")
        .map_err(|e| {
            eprintln!("Failed to load ONNX model: {}", e);
            actix_web::error::ErrorInternalServerError("Model loading error")
        })?
        .into_optimized()
        .map_err(|e| {
            eprintln!("Failed to optimize ONNX model: {}", e);
            actix_web::error::ErrorInternalServerError("Model optimization error")
        })?
        .into_runnable()
        .map_err(|e| {
            eprintln!("Failed to make model runnable: {}", e);
            actix_web::error::ErrorInternalServerError("Model initialization error")
        })?;

    let input = input_tensor.as_slice().unwrap();
    let tensor = tract_ndarray::Array::from_shape_vec((1, 3, 224, 224), input.to_vec())
        .unwrap()
        .into_tensor();

    let result = model.run(tvec!(tensor.into())).map_err(|e| {
        eprintln!("Model inference failed: {}", e);
        actix_web::error::ErrorInternalServerError("Inference error")
    })?;

    // รับค่าผลลัพธ์จากโมเดล
    let output_tensor = result[0].to_array_view::<f32>().unwrap();
    let (predicted_class, _) = output_tensor
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, &value)| (index as i64, value))
        .unwrap();

    // พิมพ์ class ที่ทำนายได้ใน terminal
    println!("Predicted class: {}", predicted_class);

    // ส่ง predicted_class ไปยัง frontend
    Ok(HttpResponse::Ok().json(predicted_class))
}
