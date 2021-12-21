#include "../inc/raw.h"

#include <fstream>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <stdexcept>


/**
 * Internal method. Replaces the default cURL write function to avoid excessive printing to `stdout`.
 *
 * @param buffer Not understood. Unused.
 * @param size Not understood.
 * @param n_mem Not understood.
 * @return Not understood.
 */
size_t write_data(char *buffer, size_t size, size_t n_mem) {
	return size * n_mem;
}


/**
 * Writes the specified URL's JSON data to the specified file.
 *
 * @param url The URL to obtain the JSON data from.
 * @param file_name The file to write to. Is saved in the `DATA_DIRECTORY`.
 */
void write_json_to_data_directory(const std::string &url, const std::string &file_name) {
	try {
		curlpp::Easy request;
		request.setOpt(new curlpp::options::Url(url));
		std::list<std::string> header;
		header.emplace_back("Content-Type: application/octet-stream");
		request.setOpt(new curlpp::options::HttpHeader(header));
		request.setOpt(new curlpp::options::WriteFunction(write_data));
		request.perform();

		std::ofstream out_file;
		out_file.open(DATA_DIRECTORY + file_name + ".json", std::ios_base::trunc);
		if (!out_file.is_open()) {
			std::cout << "Failed to open " << file_name << "!" << std::endl;
		} else {
			out_file << request;
			std::cout << "Successfully wrote to " << file_name << "!" << std::endl;
		}
	} catch (curlpp::RuntimeError &error) {
		std::cout << error.what() << std::endl;
		throw std::runtime_error("Got a runtime error!");
	} catch (curlpp::LogicError &error) {
		std::cout << error.what() << std::endl;
		throw std::runtime_error("Got a logic error!");
	}
}


Json::Value json_from_file(const std::string &file_name) {
	std::ifstream json_file(DATA_DIRECTORY + file_name + ".json", std::ifstream::binary);
	Json::Value data;
	json_file >> data;
	return data;
}


void json_to_file(const Json::Value &j_obj, const std::string &file_name, bool report) {
	Json::StyledStreamWriter writer;
	std::ofstream out_file;
	out_file.open(DATA_DIRECTORY + file_name + ".json", std::ofstream::trunc);
	if (out_file.is_open()) {
		if (report) {
			std::cout << "Writing to disk... ";
			writer.write(out_file, j_obj);
			std::cout << "Done." << std::endl;
		} else {
			writer.write(out_file, j_obj);
		}
	} else {
		throw std::runtime_error("Out file did not open!");
	}
}


void json_append(const Json::Value &j_obj, const std::string &file_name) {
	std::ifstream json_file(DATA_DIRECTORY + file_name + ".json", std::ifstream::binary);
	Json::Value data;
	try {
		json_file >> data;
	} catch (Json::Exception::exception &err) {
		json_to_file(j_obj, file_name, false);
		return;  /* no file yet, so simply write what we have */
	}
	Json::Value new_json;
	if (data.isArray() && j_obj.isArray()) {
		new_json = Json::arrayValue;
		for (const auto &entry : data) {
			new_json.append(entry);
		}
		for (const auto &entry : j_obj) {
			new_json.append(entry);
		}
	} else {
		/* continuing with the assumption that both aren't arrays */
		for (const auto &member : data.getMemberNames()) {
			new_json[member] = data[member];
		}
		for (const auto &member : j_obj.getMemberNames()) {
			new_json[member] = j_obj[member];
		}
	}
	json_to_file(new_json, file_name, false);
}
