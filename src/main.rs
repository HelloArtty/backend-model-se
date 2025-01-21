mod handlers;
mod models;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running at http://127.0.0.1:8080");

    HttpServer::new(|| {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(cors)
            .service(web::resource("/predict").route(web::post().to(handlers::predict)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
