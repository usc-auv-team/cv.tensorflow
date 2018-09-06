import cv2

def capture_video():
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_RGB)
    vc.set(cv2.CAP_PROP_FPS, 60)
    vc.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    vc.set(cv2.CAP_PROP_ISO_SPEED, 400)
    
    print(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(vc.get(cv2.CAP_PROP_MODE))
    print(vc.get(cv2.CAP_PROP_FPS))
    print(vc.get(cv2.CAP_PROP_CONVERT_RGB))
    print(vc.get(cv2.CAP_PROP_ISO_SPEED))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('/home/jaimin/Workspace/Wet Tests/output.MP4',fourcc, vc.get(cv2.CAP_PROP_FPS), (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Put the code in try-except statements
    # Catch the keyboard exception and 
    # release the camera device and 
    # continue with the rest of code.
    try:
        while(vc.isOpened()):
            # Capture frame-by-frame
            #             if vc.grab():
            #                 ret, frame = vc.retrieve()

            ret, frame = vc.read()
            if not ret:
                # Release the Video Device if ret is false
                vc.release()
                # Message to be displayed after releasing the device
                print "Released Video Resource"
                break
            # Convert the image from OpenCV BGR format to matplotlib RGB format
            # to display the image
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # the array based representation of the image will be used later in order to prepare the
            cv2.imshow('Camera Input',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            out.write(frame)

    except KeyboardInterrupt:
        # Release the Video Device
        vc.release()
        # Message to be displayed after releasing the device
        print "Released Video Resource"

    out.release()
    cv2.destroyAllWindows()
    print "Released Output Video"

capture_video()

