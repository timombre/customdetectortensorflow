/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>



#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "/usr/local/include/opencv2/objdetect.hpp"
#include "/usr/local/include/opencv2/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc.hpp"
//#include "/usr/local/include/opencv2/imgcodecs.hpp"





// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

using namespace std;

using namespace cv;


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
std::vector<std::string> ReadLabelsFile(const std::string file_name) {

    std::ifstream infile(file_name);
    std::string label;    
    std::vector<std::string> labels;

    if(infile.is_open()){
        while(std::getline(infile, label)){
            labels.push_back(label);
        }
        infile.close();
    }
    
     // size_t i =0;
     // for (int i = 0; i < labels.size(); ++i)
     // {
     //     std::cout<< labels.at(i) << " " << i+1 << std::endl;
     //     //RING_WARN("%s", labels.at(i).c_str());
     // }
    
    return labels;
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
    Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
            "' expected ", file_size, " got ",
            data.size());
    }

    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
    const int input_width, const float input_mean,
    const float input_std,
    std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
    Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
            DecodePng::Channels(wanted_channels));
    } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
            DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
         // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
         image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
           DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    // auto float_caster =
    //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    // auto resized = ResizeBilinear(
    //     root, dims_expander,
    //     Const(root.WithOpName("size"), {input_height, input_width}));


    // Subtract the mean and divide by the scale.
    // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
    //     {input_std});


    //cast to int
    //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}

// Status ReadTensorFromBuffer(const float[], const int input_height,
//                                const int input_width,
//                                std::vector<Tensor>* out_tensors) {


// auto file_reader =
//       Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

//   std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
//       {"input", input},
//   };

//   auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);



// }

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
    std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
    ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);

    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
            graph_file_name, "'");
    }

    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}




Status SaveImage(const Tensor& tensor, const string& file_path) {
    LOG(INFO) << "Saving image to " << file_path;
    CHECK(tensorflow::str_util::EndsWith(file_path, ".png"))
    << "Only saving of png files is supported.";

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string encoder_name = "encode";
    string output_name = "file_writer";

    tensorflow::Output image_encoder =
    EncodePng(root.WithOpName(encoder_name), tensor);
    tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    std::vector<Tensor> outputs;
    TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

    return Status::OK();
}

float colors[6][3] = { {255,0,255}, {0,0,255},{0,255,255},{0,255,0},{255,255,0},{255,0,0} };

float get_color_box(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}



int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
    string img(argv[1]);
    string graph ="data/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb";
    string labels ="data/mscoco_label2.txt";
    int32 input_width = 299;
    int32 input_height = 299;
    float input_mean = 0;
    float input_std = 255;
    string input_layer = "image_tensor:0";
    vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };

    bool self_test = false;
    string root_dir = "";

    //char imname[4096] = {0};

    

    

    


  //sleep(15);


  
      /* code */
  

  

  // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    LOG(ERROR) << "graph_path:" << graph_path;
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << "LoadGraph ERROR!!!!"<< load_graph_status;
        return -1;
    }

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.

    std::vector<Tensor> resized_tensors;

    string image_path = tensorflow::io::JoinPath(root_dir, img);


    


   



    Status read_tensor_status =
    ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
        input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];


    

    LOG(ERROR) <<"image shape:" << resized_tensor.shape().DebugString()<< ",len:" << resized_tensors.size() << ",tensor type:"<< resized_tensor.dtype();
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
     output_layer, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    




    //int image_width = resized_tensor.dims();
    //int image_height = 0;
    //int image_height = resized_tensor.shape()[1];

    const int image_width = resized_tensor.shape().dim_size(2);
    const int image_height = resized_tensor.shape().dim_size(1);

    LOG(ERROR) << "size:" << outputs.size() << ",image_width:" << image_width << ",image_height:" << image_height << endl;

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();

    LOG(ERROR) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();

    //tensorflow::TTypes<uint8>::Flat image_flat = resized_tensors[1].flat<tensorflow::uint8>();//resized_tensor.flat<tensorflow::uint8>()


    std::vector<std::string> lab = ReadLabelsFile(labels);

    

    //Mat image_opencv;
    Mat image_opencv(image_height, image_width, CV_8UC3, Scalar(127, 127, 127));
    //image_opencv=imread(argv[1], CV_LOAD_IMAGE_COLOR); 
    // currently bugs in opencv
    
    


    if(! image_opencv.data){
           cout <<  "Could not open or find the image" << std::endl ;
           return -1;
    }

    
       //image im = load_image_color(argv[1],0,0);
    int linewidth = std::max(1, int(image_height * .005));


    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
        if(scores(i) > 0.8)
        {
            LOG(ERROR) << i << ",score:" << scores(i) << ", label:" << lab.at(classes(i)-1) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);

            char labelstr[4096] = {0};
            strcat(labelstr, lab.at(classes(i)-1).c_str());
          

            int offset = 80*123457 % int(classes(i));
            //float rgb[3];
            //rgb[0] = get_color(2,offset,int(classes(i)));
            //rgb[1] = get_color(1,offset,int(classes(i)));
            //rgb[2] = get_color(0,offset,int(classes(i)));
            cv::Scalar color = cv::Scalar(get_color_box(0,offset,int(classes(i))), get_color_box(1,offset,int(classes(i))), get_color_box(2,offset,int(classes(i))));

            rectangle(image_opencv, cvPoint(boxes(0,i,1) * image_width, boxes(0,i,2) * image_height), cvPoint(boxes(0,i,3) * image_width, boxes(0,i,0) * image_height), color, linewidth);
            if (boxes(0,i,2) * image_height + linewidth * 3 < image_height){ //avoids printing text on top if box is too high
                putText(image_opencv, lab.at(classes(i)-1), cvPoint(boxes(0,i,1) * image_width + linewidth, boxes(0,i,0) * image_height - linewidth * 3), FONT_HERSHEY_PLAIN, linewidth * 5, color );
            }else{
                putText(image_opencv, lab.at(classes(i)-1), cvPoint(boxes(0,i,1) * image_width + linewidth, boxes(0,i,0) * image_height + linewidth * 12), FONT_HERSHEY_PLAIN, linewidth * 5, color );
            }



            //draw_box(im, boxes(0,i,1)*im.w, boxes(0,i,0)*im.h, boxes(0,i,3)*im.w, boxes(0,i,2)*im.h, rgb[0], rgb[1], rgb[2]);

          

        }
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );

    LOG(ERROR) << "Create window";
    imshow( "Display window", image_opencv); 
    LOG(ERROR) << "Display image";

    waitKey(0);





    //save_image(im, imname);




    LOG(ERROR) << "before dispose of session";

    session->Close();
    session.reset();

    LOG(ERROR) << "after dispose of session";




    //session.status.reset();
    //session.options.reset();

    //TF_CloseSession( session, status );
    //TF_DeleteSession( session, status );
    //TF_DeleteStatus( status );
    //TF_DeleteSessionOptions( options );









    //Tensor resized_tensor_boxed;
    //CreateBoxedTensor(resized_tensor, outputs[0], &resized_tensor_boxed);
    //SaveToFile(resized_tensor_boxed, "file.png");





    return 0;
}