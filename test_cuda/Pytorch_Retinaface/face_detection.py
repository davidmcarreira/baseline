


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

# ---------------------------------------------------------------------------------------

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

# ---------------------------------------------------------------------------------------

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        print("Model loaded to GPU")
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# ---------------------------------------------------------------------------------------

def detection_model(network="resnet50"):
    if network == "mobile0.25":
        cfg = cfg_mnet
        trained_model = "./weights/mobilenet0.25_Final.pth"
    elif network == "resnet50":
        cfg = cfg_re50
        trained_model = "./weights/Resnet50_Final.pth"
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, trained_model, False)
    net.eval()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") # Defines the computation device (cuda:0 => GPU)
    net = net.to(device)
    
    return net, cfg, device

# ---------------------------------------------------------------------------------------

def face_select(dets, selec_thresh, max_score=False):
    previous_area = 0
    max_area = 0
    prev_coords = np.zeros_like(dets[0])
    coords = np.zeros_like(dets[0])
    
    #print("face select: \n", dets, len(dets))
    if max_score == True: #Pre-crop that selects the face with higher score
        score = 0
        max_score = 0
        prev_crop_coords = np.zeros_like(dets[0])
        crop_coords = np.zeros_like(dets[0])

        for b in dets:
            if len(dets) == 1:
                crop_coords[:] = b
                
            if b[4]>score:
                score = b[4]
                prev_coords[:] = b
            else:
                max_score = score
                crop_coords[:] = prev_coords
        
        #crop_coords = list(map(int, crop_coords))
        #margin = 0.2 #x% around the bounding box
        #crop_height = crop_coords[3]-crop_coords[1] #ymax-ymin
        #crop_width = crop_coords[2]-crop_coords[0] #xmax-xZmin
        #crop_coords = (crop_coords[1]*(1-margin), crop_coords[0]*(1-margin), crop_height*(1+margin), crop_width*(1+margin))
        
        return crop_coords

    else:
        for b in dets:
            height = b[3]-b[1] #ymax-ymin
            width = b[2]-b[0] #xmax-xmin
            bbox_area = int(width)*int(height)


            #print(b[4])
            #if b[4]<0.001:
                #for i in range(len(coords)):
                    #coords[i] = prev_coords[i]
                #continue

            #b = list(map(int, b))
            for i in range(len(b)): # Turns into int every element from the detections except the score
                if i == 4:
                    continue
                else:
                    b[i] = int(b[i])

            #print("test", b, bbox_area, previous_area)              

            if len(dets) == 1: # Only one face present in the picture
                max_area = bbox_area
                for i in range(len(coords)):
                    coords[i] = b[i]
            else:
                if bbox_area > previous_area:
                    previous_area = bbox_area
                    for i in range(len(prev_coords)):
                        prev_coords[i] = b[i]
                else:
                    max_area = previous_area
                    for i in range(len(coords)):
                        coords[i] = prev_coords[i]


        face = np.append(coords, max_area)

        return coords

# ---------------------------------------------------------------------------------------

def crop_align(img, dets, selec_thresh, net, cfg, device, final_dir, save=False):
    '''
    b[0], b[1] is the top left corner of the bounding box
    b[2], b[3] is the lower right corner of the bounding box
    b[4] relates to the the score of the detection
    b[5], b[6] is the left eye
    b[7], b[8] is the right eye
    b[9], b[10] is the nose
    b[11], b[12] is the left of the mouth
    b[13], b[14] is the right of the mouth
    '''
    #print("dets 2 - crop align (pre): \n", dets)
    #img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    face_coords = face_select(dets, selec_thresh)
    face_coords = list(map(int, face_coords)) # Coordinates must be integers
    
    
    # -------------------- Rotation Stage ---------------------
    left_eye = (face_coords[5], face_coords[6]) # Components: (x, y)
    right_eye = (face_coords[7], face_coords[8])
    if left_eye[1] > right_eye[1]:               # Right eye is higher
        # Clock-wise rotation
        aux_point = (right_eye[0], left_eye[1])
        a = right_eye[0] - left_eye[0]
        b = right_eye[1] - aux_point[1]
        
        #cv2.circle(img_raw, left_eye, 10, (0, 255, 0), 4)
        #cv2.circle(img_raw, right_eye, 10, (0, 255, 0), 4)
        #cv2.circle(img_raw, aux_point, 10, (0, 255, 0), 4)
        
        #cv2.line(img_raw, left_eye, right_eye, (23, 23, 23), 2)
        #cv2.line(img_raw, aux_point, right_eye, (23, 23, 23), 2)
        #cv2.line(img_raw, left_eye, aux_point, (23, 23, 23), 2)
        #plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)) 
        
        theta = np.rad2deg(np.arctan(b/a)) # Angle of rotation in degrees
        #print("Right eye is higher, therefore, a clock-wise rotation of {} is applied".format(-theta))
        rotated_tensor = rotate(img.squeeze(), angle=theta, interpolation=InterpolationMode.BILINEAR, center=right_eye)

    else:                                        # Left eye is higher
        # Counter clock-wise rotation
        aux_point = (left_eye[0], right_eye[1])
        a = right_eye[0] - left_eye[0]
        b = left_eye[1] - aux_point[1]
        
        #cv2.circle(img_raw, left_eye, 10, (0, 255, 0), 4)
        #cv2.circle(img_raw, right_eye, 10, (0, 255, 0), 4)
        #cv2.circle(img_raw, aux_point, 10, (0, 255, 0), 4)
        
        #plt.imshow(img_raw)

        theta = np.rad2deg(np.arctan(b/a))
        #print("Left eye is higher, therefore, a clock-wise rotation of {} degrees is applied".format(-theta))
        rotated_tensor = rotate(img.squeeze(), angle=-theta, interpolation=InterpolationMode.BILINEAR, center=left_eye)
    
    #margin = 0.5 #x% around the bounding box
    #crop_height = face_coords[3]-face_coords[1] #ymax-ymin
    #crop_width = face_coords[2]-face_coords[0] #xmax-xZmin
    #crop_coords = (face_coords[1]*(1-margin), face_coords[0]*(1-margin), crop_height*(1+margin), crop_width*(1+margin))
    #crop_coords = list(map(int, crop_coords))
    #rotated_tensor = crop(rotated_tensor, *crop_coords)
    #plt.imshow(rotated_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(int))
    
    # -------------------- New Bounding Box computing ---------------------
    # The image is rotated, a new bbox must be generated. 
    
    # TBD: optimization by performing a preliminary crop in order to try and isolate only the relevant face
    
    loc, conf, _ = net(rotated_tensor.unsqueeze(0))  # Forward pass that gives the results <--------------
    
    im_height = rotated_tensor.shape[1]
    im_width = rotated_tensor.shape[2]
    
    resize1 = 1
    new_scale = torch.Tensor([rotated_tensor.shape[2], rotated_tensor.shape[1], rotated_tensor.shape[2], rotated_tensor.shape[1]])
    new_scale = new_scale.to(device)
    
    new_priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    new_priors = new_priorbox.forward()
    new_priors = new_priors.to(device)
    new_prior_data = new_priors.data
    
    new_boxes = decode(loc.data.squeeze(0), new_prior_data, cfg['variance'])
    new_boxes = new_boxes * new_scale / resize1
    new_boxes = new_boxes.cpu().numpy() # Tensor is moved to CPU (numpy doesn't support GPU)
    new_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    

    # Score's threshold
    confidence_threshold = 0.02 # Default value
    inds = np.where(new_scores > confidence_threshold)[0]
    new_boxes = new_boxes[inds]
    new_scores = new_scores[inds]

    # keep top-K before NMS
    top_k = 500 # Default value
    order = new_scores.argsort()[::-1][:top_k] # Extracts the indexes relating to the top scores
    new_boxes = new_boxes[order] # Array [300, 4] where in each line are the coordinates
    new_scores = new_scores[order] # Array [1, 300]
    
    
    # do NMS
    nms_threshold = 0.4 # Default value
    new_dets = np.hstack((new_boxes, new_scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(new_dets, nms_threshold)
    new_dets = new_dets[keep, :]
    
    #print("dets 3 - crop align (new dets): \n", new_dets)
    
    
    # keep top-K faster NMS
    #keep_top_k = 500 # Default value
    #new_dets = new_dets[:keep_top_k, :]
    
    #rotated_bbox = new_dets[0]
    rotated_bbox = face_select(new_dets, selec_thresh)
    #print("rotated_bbox 1", rotated_bbox)
    rotated_bbox = list(map(int, rotated_bbox))
    #print("rotated_bbox 2", rotated_bbox)
    

    
    
    # -------------------- Cropping Stage ---------------------
    crop_height = int((rotated_bbox[3]-rotated_bbox[1])*1.2) #ymax-ymin
    crop_width = int((rotated_bbox[2]-rotated_bbox[0])*1.2) #xmax-xZmin
    crop_coordinates = (rotated_bbox[1], rotated_bbox[0], crop_height, crop_width)
    cropped_tensor = crop(rotated_tensor, *crop_coordinates)
    
    #plt.imshow(cropped_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(int))
    
    
    if save == True:
        #image_array = cropped_tensor.permute(1,2,0).cpu().numpy()
        
        # Define the desired output size
        #desired_size = (180, 180)  # (height, width)

        # Calculate the padding values
        #height_diff = desired_size[0] - cropped_tensor.shape[1]
        #width_diff = desired_size[1] - cropped_tensor.shape[2]
        #top_pad = height_diff // 2
        #bottom_pad = height_diff - top_pad
        #left_pad = width_diff // 2
        #right_pad = width_diff - left_pad

        # Apply padding to the tensor
        #padded_tensor = pad(cropped_tensor, (left_pad, right_pad, top_pad, bottom_pad))
        
        final_size = (160, 160)
        #resized_tensor = resize(padded_tensor, final_size)
        resized_tensor = resize(cropped_tensor, final_size)
        
        image_array = resized_tensor.permute(1,2,0).cpu().numpy()
        
        # Convert the numpy array to BGR format (required by OpenCV)
        cropped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(final_dir, cropped_image)
    return cropped_tensor

# ---------------------------------------------------------------------------------------

# https://github.com/biubug6/Pytorch_Retinaface/
def face_detection(net, cfg, device, img, final_dir, img_raw, save=False):
    #save_image = False
    torch.set_grad_enabled(False)
    
    resize1 = 1
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    # Testing stage
    
    tic = time.time()
    loc, conf, landms = net(img)  # Forward pass that gives the results <--------------
    #print('Forward time: {:.4f}'.format(time.time() - tic))
        
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    
    boxes = boxes * scale / resize1
    boxes = boxes.cpu().numpy() # Tensor is moved to CPU (numpy doesn't support GPU)
    scores = conf.squeeze(0).data.cpu().numpy()[:,1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize1
    landms = landms.cpu().numpy()

    # Score's threshold
    confidence_threshold = 0.02 # Default value
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 500 # Default value
    order = scores.argsort()[::-1][:top_k] # Extracts the indexes relating to the top scores
    boxes = boxes[order] # Array [300, 4] where in each line are the coordinates
    landms = landms[order] # Array [300, 10]
    scores = scores[order] # Array [1, 300]

    # do NMS
    nms_threshold = 0.4 # Default value
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    keep_top_k = 750 # Default value
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    

    dets = np.concatenate((dets, landms), axis=1)
    
    #print("dets 1 - face selection: \n", dets)
    
    #for b in dets:
        #text = "{:.8f}".format(b[4])
        #b = list(map(int, b))
        #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #cx = b[0]
        #cy = b[1] + 12
        #cv2.circle(img_raw, (0, 0), 10, (0, 255, 0), 4)
        #cv2.circle(img_raw, (b[0], b[1]), 1, (255, 0, 255), 4)
        #cv2.circle(img_raw, (b[2], b[3]), 1, (255, 0, 255), 4)
        #cv2.putText(img_raw, text, (cx, cy),
                    #cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)


        #plt.imshow(img_raw)
    
    #Pre-crop
    #pre_crop_coords = face_select(dets, 0.1, pre_crop=True)
    #pre_crop_coords = list(map(int, pre_crop_coords))
    #pre_img = crop(img, *pre_crop_coords)
    
    cropped = crop_align(img, dets, 0.1, net, cfg, device, final_dir, save)
    

    return cropped, dets

# ---------------------------------------------------------------------------------------

def c_crop(image, margin_percentage):
    height, width = image.shape[:2]
    
    # Calculate the margin size in pixels
    margin_height = int(height * margin_percentage / 100)
    margin_width = int(width * margin_percentage / 100)

    # Calculate the starting coordinates for the crop
    start_y = margin_height
    start_x = margin_width

    # Calculate the target height and width after cropping
    target_height = height - 2 * margin_height
    target_width = width - 2 * margin_width

    # Perform the center crop with margin
    cropped_image = image[start_y:start_y+target_height, start_x:start_x+target_width]

    return cropped_image