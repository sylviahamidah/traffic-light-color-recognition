# Traffic Light Color Detection

A Python-based application for detecting traffic light colors (red, yellow, green) using image processing and HSV color spaces. This project processes videos with manually defined Regions of Interest (ROIs) and outputs the detected colors along with logging the details.



## Features
- Detects traffic light colors (Red, Yellow, Green) in specified ROIs.
- Processes video frames in real-time.
- Outputs processed video with detection overlays.
- Logs detection details (time, color, ROI coordinates) to a file.



## Installation and Usage
1. Clone the repository.
   ```
   git clone https://github.com/sylviahamidah/traffic-light-color-recognition.git
   cd traffic-light-color-recognition
   ```
2. Create a virtual environment and activate it.
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install the required dependencies.
   ```
   pip install -r requirements.txt
   ```
5. Run the script with the following command.
   ```
   Format: python main.py <video_path> <output_path> --log <log_file>
   Example: python main.py sample_1.mp4 outputs/output_v1.mp4 --log outputs/log_v1.txt
   ```
6. The output results in the form of log files and videos can be found in the 'outputs' folder.



## Acknowledgement
I would like to express my gratitude to the creators of the video resources used in this project. The videos are sourced from publicly available content:
### Video Sources:
1. **Sample Video 1**: [Video Traffic Light Sequence](https://youtu.be/hMzV58Y_1wE?si=IiSPZZNXtbNuDoKS)  
2. **Sample Video 2**: [Traffic Light Simpang Bakaloa, Indramayu](https://youtu.be/ZNOuyuWhMxU?si=TtYU5H8XEjE2G46T)  
3. **Sample Video 3**: [Time Lapse Lampu Lalu Lintas di Malam Hari](https://youtu.be/-GzpRPtEVBg?si=5YxmvfP7Vv78L3GX)  
   

