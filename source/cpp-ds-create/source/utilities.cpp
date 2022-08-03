/* Various utility symbols. */

#include <fstream>
#include <stdexcept>
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns JSON data loaded from a file stores in the project root
 *   `resources` directory.
 * 
 * @param file_name The name of the file to load in `resources/`. Exclude `.json`.
 * @return The JSON data.
 */
Json::Value DutchKBQADSCreate::json_loaded_from_resources_file(const std::string &file_name) {
    std::ifstream file(resources_dir / (file_name + ".json"),
                       std::ifstream::binary);
    Json::Value json;
    file >> json;
    return json;
}

/**
 * @brief Saves the supplied JSON data to the resources directory, under
 *   `file_name`.
 * 
 * @param json The JSON data to save to disk.
 * @param file_name The file in the project root's `resources` directory to save
 *   to. Exclude `.json`.
 */
void DutchKBQADSCreate::save_json_to_resources_file(const Json::Value &json,
                                                    const std::string &file_name) {
    Json::StyledStreamWriter writer;
    std::ofstream file;
    file.open(resources_dir / (file_name + ".json"),
              std::ofstream::trunc);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("JSON save file \"") +
                                 file_name +
                                 "\" won't open!");
    }
    writer.write(file, json);
}

/**
 * @brief Appends the contents of `json` to the JSON content already present in
 *   the file `file_name`, present in the project root's `resources` directory.
 * 
 * @param json The JSON to append.
 * @param file_name The name of the file in the project root's `resources`
 *   directory to which to append.
 */
void DutchKBQADSCreate::append_json_to_resources_file(const Json::Value &json,
                                                      const std::string &file_name) {
    std::ifstream file(resources_dir / (file_name + ".json"),
                       std::ifstream::binary);
    Json::Value file_json;  /* JSON already present in `file`. */
    /* Try to get the JSON content of `file`. If it fails, create the file. */
    try {
        file >> file_json;
    } catch (Json::Exception::exception &exception) {
        save_json_to_resources_file(json, file_name);
        return;
    }
    Json::Value updated_file_json;  /* `file_json` plus `json`. */
    if (file_json.isArray() && json.isArray()) {
        /* Update case 1: Append two JSON arrays. */
        updated_file_json = Json::arrayValue;
        for (const auto &entry : file_json) {
            updated_file_json.append(entry);
        }
        for (const auto &entry : json) {
            updated_file_json.append(entry);
        }
    } else {
        /* Update case 2: Approach both JSON files as objects. */
        for (const auto &member : file_json.getMemberNames()) {
            updated_file_json[member] = file_json[member];
        }
        for (const auto &member : json.getMemberNames()) {
            updated_file_json[member] = json[member];
        }
    }
    save_json_to_resources_file(updated_file_json, file_name);
}
