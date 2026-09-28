"""
This script is a prototype for testing video processing using OpenCV. 
It allows you to read a video file, display it in a window, and handle basic 
user interactions such as pausing and quitting the video playback. 

Keys while the video is playing:
- 'q': Quit the video playback.
- 'space': Pause or resume the video playback.
- 's': Save the current frame as an image file.  

"""

import sys
from pathlib import Path

import cv2

MAX_WINDOW_HEIGHT = 800

def main():
    # 1. Choose the source: a file path if given, otherwise webcam 0
    source = sys.argv[1] if len(sys.argv) > 1 else 0 # sys.argv[1] is ../footage/test.mp4
    if source != 0 and not Path(source).exists(): # In OpenCV, 0 means the first webcam
        sys.exit(f"File now found: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"OpenCV couldn't open: {source}")
    
    # 2. Print the video's details
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # cap.get(..) asks the video about itself
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # CAP_PROP... names what you're asking for
    fps = cap.get(cv2.CAP_PROP_FPS) or 30 # Webcams sometimes give 0 fps, in python 0 or 30 gives 30, otherwise will be dividing by 0 later
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source:         {source}")
    print(f"Resolution:     {width} x {height}")
    print(f"FPS:            {fps:.1f}")
    if frames > 0:
        print(f"Frames:     {frames} (~{frames / fps:.1f} s)")
    print("Keys: space = pause, s = save frame, q = quit")
    
    delay = max(1, int(1000 / fps)) # 1 second == 1000ms so each frame lasts 1000/fps, max makes sure its never 0
    paused = False # for pausing later
    frame_no = 0 # counts frames so we can display it and name save images
    frame = None # creates the variable before the loop, when paused we reuse the last fram
    
    # 3. Read and show frames one at a time
    while True:
        if not paused:
            ok, frame = cap.read() # cap.read() returns 2 values at once, ok(did it work?) and frame(the image)
            if not ok: # When video ends ok is false and break loop
                print("End of video.")
                break
            frame_no += 1
        
        # Resize for display only (the original frame is kept for saving)
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_WINDOW_HEIGHT / h)
        shown = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.putText(shown, f"Frame {frame_no}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SendIt - video test", shown)
        
        key = cv2.waitKey(0 if paused else delay) & 0xFF # waitKey waits that many milliseconds and it returns the key pressed during the wait
        if key == ord("q"): # ord turns a character into ascii
            break
        elif key == ord(" "):
            paused = not paused # toggle for pause
        elif key == ord("s"):
            out = f"frame_{frame_no:05d}.jpg" # :05d turns whole number to 5 digits so frame 42 turns to frame_00042.jpg
            cv2.imwrite(out, frame) # Saves the full-size frame
            print(f"Saved {out}")
            
        # Stop if the window was closed with the X button
        if cv2.getWindowProperty("SendIt - video test", cv2.WND_PROP_VISIBLE) <1:
            break
    
    cap.release() # Closes the video file or frees up the webcam
    cv2.destroyAllWindows() # Closes any OpenCV windows

if __name__ == "__main__":
    main()
    