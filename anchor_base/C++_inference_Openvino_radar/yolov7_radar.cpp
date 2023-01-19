#include "yolov7_radar.h"

yolo_radar::yolo_radar() {
    model = core.read_model(MODEL_PATH);
    std::shared_ptr <ov::Model> model = core.read_model(MODEL_PATH);
#ifdef DEBUG
    std::cout << "finish read model" << std::endl;
#endif
    compiled_model = core.compile_model(model, DEVICE);
#ifdef DEBUG
    std::cout << "finish compile model" << std::endl;
#endif
    auto input_port = compiled_model.input();
#ifdef DEBUG
    std::cout << "finish build inference request" << std::endl;
#endif
    infer_request = compiled_model.create_infer_request();
    input_tensor1 = infer_request.get_input_tensor(0);
#ifdef DEBUG
    std::cout << "finish creat request,now begin infer..." << std::endl;
#endif
}

cv::Mat yolo_radar::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
    //YOLO系列的letter_box算法，没有变化
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padd.push_back(padd_w);
    padd.push_back(padd_h);
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect yolo_radar::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h) {
    cv::Rect scaled_box;
    scaled_box.width = box.width / padd[2];
    scaled_box.height = box.height / padd[2];
    scaled_box.x = std::max(std::min((float) ((box.x - padd[0]) / padd[2]), (float) (raw_w - 1)), 0.f);
    scaled_box.y = std::max(std::min((float) ((box.y - padd[1]) / padd[2]), (float) (raw_h - 1)), 0.f);
    return scaled_box;
}

void yolo_radar::drawPred(int classId, float conf, cv::Rect box, cv::Mat &frame,
                          const std::vector <std::string> &classes) {
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                  cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
}

void yolo_radar::generate_proposals(int stride, const float *feat, std::vector <Object> &objects) {
    // get the results from proposals
    int feat_w = IMG_SIZE / stride;
    int feat_h = IMG_SIZE / stride;
#if ANCHOR_BIG == 0 && ANCHOR_SMALL == 2 //添加了两个小目标检测头 P2 P4
    float anchors[30] = { 4,3,  4,4,  5,4,  5,5,  7,4,  7,5,  8,4,  5,6,  8,6,  9,5,  11,6,  8,8,  12,7,  11,11,  15,9 };
#elif ANCHOR_BIG == 0 && ANCHOR_SMALL == 1 //添加了一个小目标检测头 P4
    float anchors[24] = {4, 3, 4, 4, 5, 4, 5, 5, 7, 4, 7, 5, 9, 5, 7, 7, 10, 6, 13, 7, 10, 10, 15, 10};
#elif ANCHOR_BIG == 0 && ANCHOR_SMALL == 0 //默认的情况,YOLO三个检测头
//float anchors[18] = { 4,3, 4,4, 5,4, 7,4, 6,5, 9,5, 8,8, 11,7, 13,10};
    float anchors[18] = {13, 8, 15, 9, 15, 12, 19, 15, 19, 17, 23, 19, 26, 26, 30, 23, 30, 26};
#elif ANCHOR_BIG == 1 && ANCHOR_SMALL == 0 //添加了一个大目标检测头 P64
    float anchors[24] = {4, 3, 4, 4, 5, 4, 5, 5, 7, 4, 7, 5, 9, 5, 7, 7, 10, 6, 2, 7, 10, 10, 15, 10};
#elif ANCHOR_BIG ==2 && ANCHOR_SMALL ==0 //添加了两个大目标检测头 P64 P128
float anchors[30] = { 4,3,  4,4,  5,4,  5,5,  7,4,  7,5,  8,4,  5,6,  8,6,  9,5,  11,6,  8,8,  12,7,  11,11,  15,9 };
#endif
    int anchor_group = 0;

#if ANCHOR_SMALL == 2
    if (stride == 4)
        anchor_group = 0;
#endif
#if ANCHOR_SMALL == 1
    if (stride == 4)
        anchor_group = (-1) + ANCHOR_SMALL;
#endif
    if (stride == 8) //用于切换每一个anchor层,如果有小目标，这里应该是第二个anchor层
        anchor_group = 0 + ANCHOR_SMALL;
    if (stride == 16)
        anchor_group = 1 + ANCHOR_SMALL;
    if (stride == 32)
        anchor_group = 2 + ANCHOR_SMALL;
#if ANCHOR_BIG == 1
    if (stride == 64) //最后一个anchor层
        anchor_group = 3 + ANCHOR_SMALL;
#endif
#if ANCHOR_BIG == 2
    if (stride == 128) //最后一个anchor层
        anchor_group = 4 + ANCHOR_SMALL;
#endif

    for (int anchor = 0; anchor <= 3 + ANCHOR_BIG + ANCHOR_SMALL - 1; anchor++) {
        for (int i = 0; i <= feat_h - 1; i++) {
            for (int j = 0; j <= feat_w - 1; j++) {
                float box_prob = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                                      j * (CLS_NUM + 5) + 4];
                box_prob = sigmoid(box_prob);

                // filter the bounding box with low confidence
                if (box_prob < CONF_THRESHOLD)
                    continue;
                float x = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                               j * (CLS_NUM + 5) + 0];
                float y = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                               j * (CLS_NUM + 5) + 1];
                float w = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                               j * (CLS_NUM + 5) + 2];
                float h = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                               j * (CLS_NUM + 5) + 3];

                double max_prob = 0;
                int idx = 0;

                // get the class id with maximum confidence
                for (int t = 5; t < 19; ++t) {
                    double tp = feat[anchor * feat_h * feat_w * (CLS_NUM + 5) + i * feat_w * (CLS_NUM + 5) +
                                     j * (CLS_NUM + 5) + t];
                    tp = sigmoid(tp);
                    if (tp > max_prob) {
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = std::min(box_prob * max_prob, 1.0); // 添加上一帧的保留权重
                if (cof < CONF_THRESHOLD)
                    continue;
                // 将结果转换回来
                x = (sigmoid(x) * 2 - 0.5 + j) * stride;
                y = (sigmoid(y) * 2 - 0.5 + i) * stride;
                w = pow(sigmoid(w) * 2, 2) * anchors[anchor_group * 6 + anchor * 2]; //每一层的anchor
                h = pow(sigmoid(h) * 2, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];; //每一层的anchor

                float r_x = x - w / 2;
                float r_y = y - h / 2;

                // store the results
                Object obj;
                obj.rect.x = r_x;
                obj.rect.y = r_y;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = idx - 5;
                obj.prob = cof;
                objects.push_back(obj);
            }
        }
    }
}

std::vector <yolo_radar::Object_result> yolo_radar::work(cv::Mat src_img) {
    // -------- Step 0. Set hyperparameters --------
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    cv::Mat img;
#ifdef DEBUG
    meter.start();
#endif
    std::vector<float> padd;
    cv::Mat boxed = yolo_radar::letter_box(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    auto data1 = input_tensor1.data<float>();
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int out_index = c * img_h * img_w + h * img_w + w;
                data1[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    infer_request.infer(); //开始推理
#if ANCHOR_SMALL == 2
    auto output_tensor_p2 = infer_request.get_output_tensor(0);
    const float *result_p2 = output_tensor_p2.data<const float>();
#endif
#if ANCHOR_SMALL == 1
    auto output_tensor_p4 = infer_request.get_output_tensor((-1) + ANCHOR_SMALL);
    const float *result_p4 = output_tensor_p4.data<const float>();
#endif

    auto output_tensor_p8 = infer_request.get_output_tensor(0 + ANCHOR_SMALL);
    const float *result_p8 = output_tensor_p8.data<const float>();
    auto output_tensor_p16 = infer_request.get_output_tensor(1 + ANCHOR_SMALL);
    const float *result_p16 = output_tensor_p16.data<const float>();
    auto output_tensor_p32 = infer_request.get_output_tensor(2 + ANCHOR_SMALL);
    const float *result_p32 = output_tensor_p32.data<const float>();
#if ANCHOR_BIG == 1 //64
    auto output_tensor_p64 = infer_request.get_output_tensor(3 + ANCHOR_SMALL);
    const float *result_p64 = output_tensor_p64.data<const float>();
#endif
#if ANCHOR_BIG == 2 //128
    auto output_tensor_p128 = infer_request.get_output_tensor(4 + ANCHOR_SMALL);
    const float *result_p128 = output_tensor_p2.data<const float>();
#endif
    std::vector <Object> proposals;

#if ANCHOR_SMALL == 2
    std::vector<Object> objects2;
    generate_proposals(2, result_p2, objects2);
    proposals.insert(proposals.end(), objects2.begin(), objects2.end());
#endif
#if ANCHOR_SMALL >= 1
    std::vector<Object> objects4;
    generate_proposals(4, result_p4, objects4);
    proposals.insert(proposals.end(), objects4.begin(), objects4.end());
#endif
    std::vector <Object> objects8;
    std::vector <Object> objects16;
    std::vector <Object> objects32;
    yolo_radar::generate_proposals(8, result_p8, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    yolo_radar::generate_proposals(16, result_p16, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    yolo_radar::generate_proposals(32, result_p32, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());

#if ANCHOR_BIG == 1
    std::vector<Object> objects64;
    generate_proposals(64, result_p64, objects64);
    proposals.insert(proposals.end(), objects64.begin(), objects64.end());
#endif
#if ANCHOR_BIG == 2
    std::vector<Object> objects128;
    generate_proposals(128, result_p128, objects128);
    proposals.insert(proposals.end(), objects128.begin(), objects128.end());
#endif

    std::vector<int> classIds; //所有被选中bbox的ID
    std::vector<float> confidences; //所有被选中bbox的置信度
    std::vector <cv::Rect> boxes; //所有被选中bbox
    for (size_t i = 0; i < proposals.size(); i++) {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
    }
    std::vector<int> picked; //被选出的置信框
    std::vector<float> picked_useless;
    std::vector <Object_result> object_result;
    cv::dnn::softNMSBoxes(boxes, confidences, picked_useless, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    for (size_t i = 0; i < picked.size(); i++) {
        cv::Rect scaled_box = scale_box(boxes[picked[i]], padd, src_img.cols, src_img.rows);
        Object_result obj;
        obj.bbox = scaled_box;
        obj.label = classIds[picked[i]];
        obj.prob = confidences[picked[i]];
        object_result.push_back(obj);
#ifdef VIDEOS
        drawPred(classIds[picked[i]], confidences[picked[i]], scaled_box, src_img, class_names);
#endif
    }
#ifdef DEBUG
    meter.stop();
    printf("Time: %f\n", meter.getTimeMilli());
    meter.reset();
#endif
#ifdef VIDEOS
    cv::imshow("Inference frame", src_img);
    cv::waitKey(1);
#endif
    return object_result;
}