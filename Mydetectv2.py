import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np 
from numpy import random
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def process(xyxy, im0, im_list,xyxy_list ):  
    #xyxy is the postion of the bounding box, the top left and bottom right corner
                        mask = np.zeros(im0.shape, dtype=im0.dtype)
                        xyxycpu = np.array(torch.tensor(xyxy, device= 'cpu'))
                        xyxycpu = xyxycpu.astype(np.int)
                        xyxy_temp = xyxycpu
                        xyxy_list.append(xyxy_temp.tolist())
                        print('xyxycpu =', xyxycpu)
                        slice_im = im0[xyxycpu[1]:xyxycpu[3],xyxycpu[0]:xyxycpu[2]]
                        temp_img = slice_im
                        im_list.append(temp_img.tolist())
                        print(  "list range = ", len(im_list))
                        
                            # print("Im list = ", im_list[i])
                        print('image shape = ', slice_im.shape[0],slice_im.shape[1])
                        return slice_im

def detect(save_img=False): 
    source, weights, contact_weights, view_img, save_txt, imgsz, trace =  opt.source, opt.weights, opt.contact_weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    # I add
    
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    contact_model = attempt_load(contact_weights,map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

   
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

       
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            #initialize image list
            im_list = []
            xyxy_list = []
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #I add here
                    print('xyxy=',torch.tensor(xyxy))
                    #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    #print('xywh=',xywh)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print("xywh=",xywh)
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:
                        print("in save process")
                        slice_im = process(xyxy,im0,im_list,xyxy_list)

                    #接下來要做的是把slice下來的image也就是im_list裡面的照片依照他的xyxy位置放到一張原圖大小的白色image
                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey()  # 1 millisecond

            
            # # Create a black image with the same size as big_image
            # mask = np.zeros(big_image.shape, dtype=big_image.dtype)

            # # Paste small_image onto the black image at position (x, y)
            # mask[y:y+small_image.shape[0], x:x+small_image.shape[1]] = small_image

            # # Use cv2.copyTo to paste the small_image onto big_image
            # cv2.copyTo(mask, small_image, mask)

            # # Composite the two images together using cv2.addWeighted
            # alpha = 1.0  # Weighting factor for small_image
            # beta = 1.0 - alpha  # Weighting factor for big_image
            # output = cv2.addWeighted(big_image, beta, mask, alpha, 0.0)
            # Save results (image with detections)
            if save_img:
                mask = np.ones(im0.shape,dtype=im0.dtype)
                mask[:,:] = [255,255,255]       #initiate to all white background
                print("mask shape = ", mask.shape)
                for i in range(len(im_list)):
                    print("xyxy in for  = ", xyxy_list[i-1], "image i shape = ", len(im_list[i-1]))
                    cur_xyxy = xyxy_list[i-1]
                    mask[cur_xyxy[1]:cur_xyxy[3],cur_xyxy[0]:cur_xyxy[2]] = im_list[i-1]
                    #mask = torch.from_numpy(mask).to(device)
                    mask_tensor = torch.tensor(mask)
                # connect second model
                with torch.no_grad():
                    contact_pred = contact_model(mask_tensor, augment= opt.augment)[0]
                contact_pred = non_max_suppression(contact_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                for i, contact_det in enumerate(contact_pred):
                    if len(contact_det):
                # Rescale boxes from img_size to im0 size
                        contact_det[:, :4] = scale_coords(img.shape[2:], contact_det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                    #I add here
                            print('xyxy 2nd=',torch.tensor(xyxy))
                if dataset.mode == 'image':
                    
                    cv2.imwrite(save_path,mask)
                        

                    
                    
                    #crop_im = im0.crop((xywh[0],xywh[1],xywh[2],xywh[3]))
                    
                    #cv2.imwrite(save_path, im0)
                    #cv2.imwrite(save_path,crop_im)
                    # for sliceim in slice_im:
                        # cv2.imshow(str(p),sliceim)
                        # cv2.waitKey()
                    if slice_im.shape[0] > 0 and slice_im.shape[1] > 0:
                        print('sliceim shape  = ' ,slice_im.shape[0],slice_im.shape[1])
                        # cv2.imwrite(save_path,slice_im)
                    else:
                        print("Is empty")
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, mask.shape[1], mask.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(mask)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--contact-weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


