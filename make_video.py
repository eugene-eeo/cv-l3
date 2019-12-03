import cv2
import glob


out = cv2.VideoWriter('assignment.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, (1024, 934))
files = sorted(glob.glob('res/*.png'))
for filename in files:
    print(filename)
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    out.write(img)
out.release()
