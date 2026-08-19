#include "sam3.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

#ifdef _WIN32
#define MODEL_LAB_POPEN _popen
#define MODEL_LAB_PCLOSE _pclose
#define MODEL_LAB_POPEN_MODE "rb"
#else
#define MODEL_LAB_POPEN popen
#define MODEL_LAB_PCLOSE pclose
#define MODEL_LAB_POPEN_MODE "r"
#endif

struct cli_args {
    std::string mode;
    std::map<std::string, std::vector<std::string>> values;
    std::map<std::string, bool> flags;
};

std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (const char ch : value) {
        if (ch == '\'') quoted += "'\\''";
        else quoted += ch;
    }
    quoted += "'";
    return quoted;
}

class sequential_video_reader {
public:
    sequential_video_reader(const std::string & path, int width, int height)
        : width_(width), height_(height) {
        if (width_ <= 0 || height_ <= 0) throw std::runtime_error("Invalid video dimensions");
        std::ostringstream command;
        command << "ffmpeg -nostdin -loglevel error -noautorotate -i " << shell_quote(path)
                << " -map 0:v:0 -an -sn -dn -c:v rawvideo -threads:v 1"
                << " -f rawvideo -pix_fmt rgb24 pipe:1";
        pipe_ = MODEL_LAB_POPEN(command.str().c_str(), MODEL_LAB_POPEN_MODE);
        if (!pipe_) throw std::runtime_error("Failed to start the FFmpeg frame decoder");
    }

    sequential_video_reader(const sequential_video_reader &) = delete;
    sequential_video_reader & operator=(const sequential_video_reader &) = delete;

    ~sequential_video_reader() {
        if (pipe_) MODEL_LAB_PCLOSE(pipe_);
    }

    sam3_image frame(int requested_index) {
        if (requested_index < next_index_) {
            throw std::runtime_error("The sequential video reader cannot seek backwards");
        }
        sam3_image image;
        while (next_index_ <= requested_index) {
            image.width = width_;
            image.height = height_;
            image.channels = 3;
            image.data.resize(static_cast<size_t>(width_) * height_ * 3);
            size_t offset = 0;
            while (offset < image.data.size()) {
                const size_t count = std::fread(
                    image.data.data() + offset,
                    1,
                    image.data.size() - offset,
                    pipe_
                );
                if (count == 0) {
                    image.data.clear();
                    return image;
                }
                offset += count;
            }
            ++next_index_;
        }
        return image;
    }

private:
    FILE * pipe_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int next_index_ = 0;
};

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

cli_args parse_cli(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("Expected mode: image or video");
    cli_args parsed;
    parsed.mode = argv[1];
    for (int index = 2; index < argc; ++index) {
        std::string key = argv[index];
        if (key.rfind("--", 0) != 0) throw std::runtime_error("Unexpected argument: " + key);
        if (key == "--no-gpu" || key == "--multimask") {
            parsed.flags[key] = true;
            continue;
        }
        if (index + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        parsed.values[key].push_back(argv[++index]);
    }
    return parsed;
}

std::string one(const cli_args & args, const std::string & key, const std::string & fallback = "") {
    auto found = args.values.find(key);
    return found == args.values.end() || found->second.empty() ? fallback : found->second.back();
}

std::vector<std::string> many(const cli_args & args, const std::string & key) {
    auto found = args.values.find(key);
    return found == args.values.end() ? std::vector<std::string>{} : found->second;
}

std::vector<float> parse_numbers(const std::string & value, size_t count) {
    std::vector<float> result;
    std::stringstream stream(value);
    std::string field;
    while (std::getline(stream, field, ',')) result.push_back(std::stof(field));
    if (result.size() != count) throw std::runtime_error("Bad coordinate value: " + value);
    return result;
}

sam3_point parse_point(const std::string & value) {
    auto numbers = parse_numbers(value, 2);
    return {numbers[0], numbers[1]};
}

sam3_box parse_box(const std::string & value) {
    auto numbers = parse_numbers(value, 4);
    return {numbers[0], numbers[1], numbers[2], numbers[3]};
}

int integer(const cli_args & args, const std::string & key, int fallback) {
    const auto value = one(args, key);
    return value.empty() ? fallback : std::stoi(value);
}

float number(const cli_args & args, const std::string & key, float fallback) {
    const auto value = one(args, key);
    return value.empty() ? fallback : std::stof(value);
}

sam3_pvs_params parse_object(const std::string & value, int * frame = nullptr, int * id = nullptr) {
    sam3_pvs_params params;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (item.empty()) continue;
        const auto separator = item.find(':');
        if (separator == std::string::npos) throw std::runtime_error("Bad object item: " + item);
        const auto key = item.substr(0, separator);
        const auto body = item.substr(separator + 1);
        if (key == "p") params.pos_points.push_back(parse_point(body));
        else if (key == "n") params.neg_points.push_back(parse_point(body));
        else if (key == "b") { params.box = parse_box(body); params.use_box = true; }
        else if (key == "frame" && frame) *frame = std::stoi(body);
        else if (key == "id" && id) *id = std::stoi(body);
        else throw std::runtime_error("Unknown object item: " + key);
    }
    return params;
}

struct refinement {
    int frame = 0;
    int id = -1;
    std::vector<sam3_point> positive;
    std::vector<sam3_point> negative;
};

refinement parse_refinement(const std::string & value) {
    int frame = 0;
    int id = -1;
    auto prompts = parse_object(value, &frame, &id);
    if (id < 0) throw std::runtime_error("Refinement requires id:<object id>");
    return {frame, id, prompts.pos_points, prompts.neg_points};
}

void write_detection(
    std::ostream & output,
    const sam3_detection & detection,
    const std::string & output_dir,
    int frame_index,
    int item_index,
    bool first
) {
    const std::string mask_name = "frame_" + std::to_string(frame_index) + "_mask_" +
                                  std::to_string(item_index) + ".png";
    if (!sam3_save_mask(detection.mask, output_dir + "/" + mask_name)) {
        throw std::runtime_error("Failed to save mask " + mask_name);
    }
    if (!first) output << ',';
    output << "{\"box\":[" << detection.box.x0 << ',' << detection.box.y0 << ','
           << detection.box.x1 << ',' << detection.box.y1 << "],\"score\":" << detection.score
           << ",\"iou_score\":" << detection.iou_score << ",\"instance_id\":"
           << detection.instance_id << ",\"mask\":\"" << json_escape(mask_name) << "\"}";
}

void write_frame(
    std::ostream & output,
    const sam3_result & result,
    const std::string & output_dir,
    int frame_index,
    bool first_frame
) {
    if (!first_frame) output << ',';
    output << "{\"frame_index\":" << frame_index << ",\"detections\":[";
    for (size_t index = 0; index < result.detections.size(); ++index) {
        write_detection(output, result.detections[index], output_dir, frame_index,
                        static_cast<int>(index), index == 0);
    }
    output << "]}";
}

sam3_params model_params(const cli_args & args) {
    sam3_params params;
    params.model_path = one(args, "--model");
    params.n_threads = integer(args, "--threads", 4);
    params.use_gpu = !args.flags.count("--no-gpu");
    params.encode_img_size = integer(args, "--encode-img-size", 0);
    if (params.model_path.empty()) throw std::runtime_error("--model is required");
    return params;
}

std::ofstream manifest_start(
    const cli_args & args,
    const std::string & source,
    const std::string & prompt_mode,
    const std::string & prompt,
    int width,
    int height,
    float fps,
    const std::string & model_name
) {
    const std::string output_dir = one(args, "--output-dir");
    if (output_dir.empty()) throw std::runtime_error("--output-dir is required");
    std::ofstream output(output_dir + "/manifest.json");
    if (!output) throw std::runtime_error("Cannot create manifest.json");
    output << std::setprecision(8)
           << "{\"schema_version\":1,\"runtime\":\"sam3.cpp\",\"model\":\""
           << json_escape(model_name) << "\",\"prompt_mode\":\"" << json_escape(prompt_mode)
           << "\",\"prompt\":\"" << json_escape(prompt) << "\",\"source\":\""
           << json_escape(source) << "\",\"width\":" << width << ",\"height\":" << height;
    if (fps > 0) output << ",\"fps\":" << fps;
    output << ",\"frames\":[";
    return output;
}

int run_image(const cli_args & args) {
    const auto params = model_params(args);
    const std::string image_path = one(args, "--image");
    const std::string output_dir = one(args, "--output-dir");
    const std::string prompt_mode = one(args, "--prompt-mode", "text");
    const std::string prompt = one(args, "--text");
    if (image_path.empty()) throw std::runtime_error("--image is required");

    auto model = sam3_load_model(params);
    if (!model) throw std::runtime_error("Failed to load model");
    auto state = sam3_create_state(*model, params);
    if (!state) throw std::runtime_error("Failed to create state");
    const auto image = sam3_load_image(image_path);
    if (image.data.empty()) throw std::runtime_error("Failed to load image");
    if (!sam3_encode_image(*state, *model, image)) throw std::runtime_error("Image encoding failed");

    sam3_result result;
    if (prompt_mode == "text") {
        sam3_pcs_params pcs;
        pcs.text_prompt = prompt;
        pcs.score_threshold = number(args, "--score-threshold", 0.35f);
        pcs.nms_threshold = number(args, "--nms-threshold", 0.1f);
        for (const auto & value : many(args, "--pos-exemplar")) pcs.pos_exemplars.push_back(parse_box(value));
        for (const auto & value : many(args, "--neg-exemplar")) pcs.neg_exemplars.push_back(parse_box(value));
        result = sam3_segment_pcs(*state, *model, pcs);
    } else if (prompt_mode == "visual") {
        sam3_pvs_params pvs;
        for (const auto & value : many(args, "--positive")) pvs.pos_points.push_back(parse_point(value));
        for (const auto & value : many(args, "--negative")) pvs.neg_points.push_back(parse_point(value));
        const auto box_value = one(args, "--box");
        if (!box_value.empty()) { pvs.box = parse_box(box_value); pvs.use_box = true; }
        pvs.multimask = args.flags.count("--multimask") != 0;
        result = sam3_segment_pvs(*state, *model, pvs);
    } else {
        throw std::runtime_error("--prompt-mode must be text or visual");
    }

    auto output = manifest_start(args, image_path, prompt_mode, prompt, image.width, image.height, 0, "SAM 3 Q8_0");
    write_frame(output, result, output_dir, 0, true);
    output << "]}";
    return 0;
}

int run_video(const cli_args & args) {
    const auto params = model_params(args);
    const std::string video_path = one(args, "--video");
    const std::string output_dir = one(args, "--output-dir");
    const std::string prompt_mode = one(args, "--prompt-mode", "text");
    const std::string prompt = one(args, "--text");
    if (video_path.empty()) throw std::runtime_error("--video is required");
    const auto info = sam3_get_video_info(video_path);
    if (info.n_frames <= 0) throw std::runtime_error("Failed to inspect video; is ffmpeg installed?");

    auto model = sam3_load_model(params);
    if (!model) throw std::runtime_error("Failed to load model");
    auto state = sam3_create_state(*model, params);
    if (!state) throw std::runtime_error("Failed to create state");
    const int start_frame = std::max(0, integer(args, "--start-frame", 0));
    const int max_frames = integer(args, "--max-frames", 0);
    const int end_frame = max_frames > 0 ? std::min(info.n_frames, start_frame + max_frames) : info.n_frames;
    sequential_video_reader video_reader(video_path, info.width, info.height);
    const std::vector<refinement> refinements = [&]() {
        std::vector<refinement> values;
        for (const auto & item : many(args, "--refine")) values.push_back(parse_refinement(item));
        return values;
    }();

    auto output = manifest_start(args, video_path, prompt_mode, prompt, info.width, info.height, info.fps, "SAM 3 Q8_0");
    bool first_frame = true;
    if (prompt_mode == "text") {
        sam3_video_params video_params;
        video_params.text_prompt = prompt;
        video_params.score_threshold = number(args, "--score-threshold", 0.35f);
        video_params.nms_threshold = number(args, "--nms-threshold", 0.1f);
        video_params.assoc_iou_threshold = number(args, "--assoc-iou", 0.1f);
        video_params.max_keep_alive = integer(args, "--max-keep-alive", 30);
        video_params.recondition_every = integer(args, "--recondition-every", 16);
        video_params.fill_hole_area = integer(args, "--fill-hole-area", 16);
        auto tracker = sam3_create_tracker(*model, video_params);
        if (!tracker) throw std::runtime_error("Failed to create text tracker");
        for (int frame_index = start_frame; frame_index < end_frame; ++frame_index) {
            const auto frame = video_reader.frame(frame_index);
            if (frame.data.empty()) throw std::runtime_error("Failed to decode frame " + std::to_string(frame_index));
            auto result = sam3_track_frame(*tracker, *state, *model, frame);
            for (const auto & refine : refinements) {
                if (refine.frame == frame_index)
                    sam3_refine_instance(*tracker, *state, *model, refine.id, refine.positive, refine.negative);
            }
            write_frame(output, result, output_dir, frame_index, first_frame);
            first_frame = false;
        }
    } else if (prompt_mode == "visual") {
        sam3_visual_track_params video_params;
        video_params.assoc_iou_threshold = number(args, "--assoc-iou", 0.1f);
        video_params.max_keep_alive = integer(args, "--max-keep-alive", 30);
        video_params.recondition_every = integer(args, "--recondition-every", 16);
        video_params.fill_hole_area = integer(args, "--fill-hole-area", 16);
        auto tracker = sam3_create_visual_tracker(*model, video_params);
        if (!tracker) throw std::runtime_error("Failed to create visual tracker");
        const auto first = video_reader.frame(start_frame);
        if (first.data.empty() || !sam3_encode_image(*state, *model, first))
            throw std::runtime_error("Failed to encode initial frame");
        sam3_result initial;
        for (const auto & value : many(args, "--object")) {
            auto prompts = parse_object(value);
            const int id = sam3_tracker_add_instance(*tracker, *state, *model, prompts);
            auto object_result = sam3_segment_pvs(*state, *model, prompts);
            if (!object_result.detections.empty()) {
                object_result.detections[0].instance_id = id;
                object_result.detections[0].mask.instance_id = id;
                initial.detections.push_back(std::move(object_result.detections[0]));
            }
        }
        write_frame(output, initial, output_dir, start_frame, true);
        first_frame = false;
        for (int frame_index = start_frame + 1; frame_index < end_frame; ++frame_index) {
            const auto frame = video_reader.frame(frame_index);
            if (frame.data.empty()) throw std::runtime_error("Failed to decode frame " + std::to_string(frame_index));
            auto result = sam3_propagate_frame(*tracker, *state, *model, frame);
            for (const auto & refine : refinements) {
                if (refine.frame == frame_index)
                    sam3_refine_instance(*tracker, *state, *model, refine.id, refine.positive, refine.negative);
            }
            write_frame(output, result, output_dir, frame_index, first_frame);
            first_frame = false;
        }
    } else {
        throw std::runtime_error("--prompt-mode must be text or visual");
    }
    output << "]}";
    return 0;
}

void usage() {
    std::cerr << "sam3_bridge image --model FILE --image FILE --output-dir DIR [prompts]\n"
              << "sam3_bridge video --model FILE --video FILE --output-dir DIR [prompts]\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = parse_cli(argc, argv);
        if (args.mode == "image") return run_image(args);
        if (args.mode == "video") return run_video(args);
        usage();
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "sam3_bridge: " << error.what() << '\n';
        return 1;
    }
}
