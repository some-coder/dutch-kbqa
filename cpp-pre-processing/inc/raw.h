#ifndef DOWNLOAD_RAW_H
#define DOWNLOAD_RAW_H


#include <cstdlib>
#include <string>
#include <json/json.h>


#define TRAIN_JSON "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/train.json"
#define TEST_JSON "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test.json"

#define DATA_DIRECTORY "../data/"


size_t write_data(char *buffer, size_t size, size_t n_mem);
void write_json_to_data_directory(const std::string &url, const std::string &file_name);
Json::Value json_from_file(const std::string &file_name);
void json_to_file(const Json::Value &j_obj, const std::string &file_name, bool report = false);
void json_append(const Json::Value &j_obj, const std::string &file_name);


#endif // DOWNLOAD_RAW_H
