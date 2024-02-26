use anyhow::Result;
use opencv::{
    prelude::*,
    videoio,
    highgui,
    imgproc,
    core,
};

fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let mut frame = Mat::default();

    let mut hsv = Mat::default();
    let mut green_mask = Mat::default();
    let mut filtered = Mat::default();
    let mut gauss_gray = Mat::default();
    let mut circles = core::Vector::<core::Vec3f>::new();

    while videoio::VideoCapture::is_opened(&cam)? {
        cam.read(&mut frame)?;
        imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        // Define the range for green color in HSV
        let lower_green = opencv::core::Scalar::new(35.0, 90.0, 90.0, 0.0); // Adjust these values for your specific green
        let upper_green = opencv::core::Scalar::new(85.0, 255.0, 255.0, 0.0); // Adjust these values for your specific green

        // Create a mask for green color
        core::in_range(&hsv, &lower_green, &upper_green, &mut green_mask)?;

        // Apply the mask to get only green parts
        core::bitwise_and(&frame, &frame, &mut filtered, &green_mask)?;

        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(&filtered, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Apply Gaussian blur
        imgproc::gaussian_blur(&gray, &mut gauss_gray, opencv::core::Size::new(9, 9), 2.0, 2.0, opencv::core::BORDER_DEFAULT)?;

        // Hough Circle Detection
        let dp = 1.0;
        let min_dist = 100.0;
        let param1 = 100.0;
        let param2 = 30.0;
        let min_radius = 20;
        let max_radius = 200;

        imgproc::hough_circles(&gauss_gray, &mut circles, imgproc::HOUGH_GRADIENT, dp, min_dist, param1, param2, min_radius, max_radius)?;

        // Draw circles
        for circle in circles.iter() {
            let center = opencv::core::Point::new(circle[0] as i32, circle[1] as i32);
            let radius = circle[2] as i32;

            // Draw the circle outline
            imgproc::circle(&mut frame, center, radius, opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), 3, 8, 0)?;
        }

        highgui::imshow("window", &frame)?;
        let key = highgui::wait_key(1)?;
        if key == 113 {
            break;
        }
    }

    Ok(())
}
