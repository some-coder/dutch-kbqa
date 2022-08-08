/* Symbols for retrieving labels for WikiData entities and properties. */

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>
#include <chrono>
#include <thread>
#include "tasks/label-entities-properties.hpp"
#include "tasks/collect-entities-properties.hpp"
#include "utilities.hpp"

using namespace DutchKBQADSCreate;

/**
 * @brief Returns the set of entities and properties present in the
 *   question-to-entities-and-properties map of `split`.
 *
 * @param split The split to retrieve the unique entities and properties of.
 * @return The set.
 */
std::set<std::string> DutchKBQADSCreate::unique_entities_and_properties_of_split(const LCQuADSplit &split) {
    Json::Value json = loaded_question_entities_properties_map(split);
    std::set<std::string> ent_prp_set;
    for (const auto &member : json.getMemberNames()) {
        for (const auto &ent_or_prp : json[member]) {
            ent_prp_set.insert(ent_or_prp.asString());
        }
    }
    return ent_prp_set;
}

/**
 * @brief Returns the file name of the WikiData entity-and-property labels
 *   file.
 *
 * @param split The LC-QuAD 2.0 dataset split for which to get the file name.
 * @param language The natural language in which the labels are expressed.
 * @return The file name.
 */
std::string entity_and_property_labels_file_name(const LCQuADSplit &split,
                                                 const NaturalLanguage &language) {
    return string_from_lc_quad_split(split) +
           "-" +
           string_from_natural_language(language) +
           "-entity-property-labels";
}

/**
 * @brief Saves the entity-and-property labels to disk.
 *
 * This function appends the supplied entities and properties if a file storing
 * WikiData entities and properties already exists on disk; if said file does
 * not yet exist, this function will also create the file before writing the
 * labels.
 *
 * @param json The WikiData labels of entities and properties, stored as a JSON
 *   object.
 * @param split The LC-QuAD 2.0 dataset split of which `json` stores the labels.
 * @param language The natural language of the labels of `json`.
 */
void DutchKBQADSCreate::save_entity_and_property_labels(const Json::Value &json,
                                                        const LCQuADSplit &split,
                                                        const NaturalLanguage &language) {
    const std::string relative_path = std::string("supplements/") +
                                      entity_and_property_labels_file_name(split, language);
    if (dataset_file_exists(relative_path + ".json")) {
        append_json_to_dataset_file(json, relative_path);
    } else {
        save_json_to_dataset_file(json, relative_path);
    }
}

/**
 * @brief Returns the required entity-and-property labels file loaded from
 *   disk.
 *
 * @param split The LC-QuAD 2.0 dataset split to target.
 * @param language The natural language to target.
 * @return The loaded labels file, provided that it exists on disk. If not, an
 *   empty JSON object is returned instead.
 */
Json::Value DutchKBQADSCreate::loaded_entity_and_property_labels(const LCQuADSplit &split,
                                                                 const NaturalLanguage &language) {
    const std::string relative_path = std::string("supplements/") +
                                      entity_and_property_labels_file_name(split, language);
    if (dataset_file_exists(relative_path + ".json")) {
        return json_loaded_from_dataset_file(relative_path);
    } else {
        return {};  /* return an empty JSON object if it isn't found */
    }
}

/**
 * @brief Returns the WikiData entities and properties that have not yet been
 *   labelled in the required natural language.
 *
 * Internally, this function checks whether labels already exist for the given
 * split-natural language combination by opening a pre-specified JSON file.
 *
 * @param split The LC-QuAD 2.0 dataset split for which labelling of entities
 *   and properties needs to be performed.
 * @param language The natural language in which the labelling needs to be
 *   performed.
 * @return The set of WikiData entities and properties that still require
 *   labelling.
 */
std::set<std::string> DutchKBQADSCreate::entities_and_properties_requiring_labeling(const LCQuADSplit &split,
                                                                                    const NaturalLanguage &language) {
    Json::Value current_json = loaded_entity_and_property_labels(split, language);
    const std::set<std::string> ent_prp_total = unique_entities_and_properties_of_split(split);
    std::set<std::string> ent_prp_labelled;
    for (const auto &ent_or_prp : current_json.getMemberNames()) {
        ent_prp_labelled.insert(ent_or_prp);
    }
    std::cout << "" << "(Found " << ent_prp_labelled.size() << " already-labelled symbols.)" << std::endl;
    std::set<std::string> ent_prp_to_label;
    std::set_difference(ent_prp_total.begin(), ent_prp_total.end(),
                        ent_prp_labelled.begin(), ent_prp_labelled.end(),
                        std::inserter(ent_prp_to_label, ent_prp_to_label.begin()));
    return ent_prp_to_label;
}

/**
 * @brief Partitions the provided set of entities and properties into (mostly)
 *   `part_size`-sized sets.
 *
 * @param ent_prp_set The entities and properties to partition.
 * @param part_size The size that the partition's parts should maximally have.
 *   All or all but one of the parts will be of this size. The very last part
 *   may be of a smaller size than `part_size`, but its size is guaranteed to
 *   be at least 1.
 * @return The parts of the partition: smaller sets of minimally 1 and
 *   maximally `part_size` size.
 */
ent_prp_partitioning DutchKBQADSCreate::entity_property_partitioning(const std::set<std::string> &ent_prp_set,
                                                                     int part_size) {
    if ((part_size < 0) || (static_cast<size_t>(part_size) > ent_prp_set.size())) {
        throw std::invalid_argument(std::string("Part size ") +
                                    std::to_string(part_size) +
                                    " is inappropriate for entity and" +
                                    " property set of length " +
                                    std::to_string(ent_prp_set.size()) +
                                    ".");
    }
    ent_prp_partitioning partitioning;
    int count = 0;
    for (const auto &ent_or_prp : ent_prp_set) {
        if ((count % part_size) == 0) {
            partitioning.emplace_back();  /* begin a new part */
        }
        partitioning[partitioning.size() - 1].insert(ent_or_prp);
        count++;
    }
    return partitioning;
}

/**
 * @brief Returns a WikiData SPARQL query for obtaining labels associated with
 *   the entity or property `ent_or_prp`.
 *
 * @param ent_or_prp The WikiData entity or property to get the labels of.
 * @param language The natural language in which the labels should be
 *   expressed.
 * @param indent_level The indentation level of the query. Must be
 *   non-negative.
 * @return The query.
 */
std::string wikidata_labelling_query_for_one_entity_or_property(const std::string &ent_or_prp,
                                                                const NaturalLanguage &language,
                                                                int indent_level) {
    if (indent_level < 0) {
        throw std::invalid_argument("Indent level must be non-negative!");
    }
    const char tab = '\t';
    const char nl = '\n';
    const std::string sparql_nl(" .\n");
    const std::string indent(indent_level, tab);
    return indent + "SELECT DISTINCT ?id ?label WHERE {" + nl +
           indent + tab + "BIND(\"" + ent_or_prp + "\" AS ?id)" + sparql_nl +
           indent + tab + '{' + nl +
           indent + tab + tab + "wd:" + ent_or_prp + " rdfs:label ?label" + sparql_nl +
           indent + tab + "} UNION {" + nl +
           indent + tab + tab + "wd:" + ent_or_prp + " skos:altLabel ?label" + sparql_nl +
           indent + tab + '}' + nl +
           indent + tab + "FILTER(LANG(?label) = \"" + string_from_natural_language(language) + "\")" + sparql_nl +
           indent + "}" + nl;
}

/**
 * @brief Returns a WikiData SPARQL query for obtaining labels associated with
 *   multiple entities and properties, collected in `ent_prp_part`.
 *
 * @param ent_prp_part The partition part containing one or more entities and
 *   properties.
 * @param language A natural language to express the labels in.
 * @return The query.
 */
std::string wikidata_labelling_query_for_entities_and_properties(const std::set<std::string> &ent_prp_part,
                                                                 const NaturalLanguage &language) {
    std::string query = std::string("SELECT ?id ?label WHERE {\n");
    int count = 0;
    for (const auto &ent_or_prp : ent_prp_part) {
        query += "\t{\n";
        query += wikidata_labelling_query_for_one_entity_or_property(ent_or_prp,
                                                                     language,
                                                                     2);
        query += "\t}\n";
        if (static_cast<size_t>(count) != ent_prp_part.size() - 1) {
            query += "\tUNION\n";
        }
        count++;
    }
    query += "}";
    return query;
}

/**
 * @brief Encodes a string for usage in a URL.
 *
 * This function's implementation is derived from a StackOverflow answer given
 * here:
 *   https://stackoverflow.com/a/17708801
 *
 * @param str The string to encode for use in URLs.
 * @return The encoded string.
 */
std::string url_encoded_string(const std::string &str) {
    std::ostringstream encoded;
    encoded.fill('0');
    encoded << std::hex;
    for (char c : str) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            /* Pass unreserved characters as-is. See RFC 3986, section 2.3. */
            encoded << c;
        } else {
            /* Percent-encode reserved characters. See RFC 3986, section 2.2. */
            encoded << std::uppercase;
            encoded << '%' << std::setw(2) << int((unsigned char) c);
            encoded << std::nouppercase;
        }
    }
    return encoded.str();
}

const std::string wikidata_query_service_url = "https://query.wikidata.org/";

/**
 * @brief Writes binary data items from a source location, `ptr`, to a stream
 *   handled by a caller, doing this without printing anything to standard
 *   output.
 *
 * Normally, this `fwrite`-like function has a fourth parameter: `FILE *stream`,
 * pointing to a stream to which the data items should be written. This method,
 * being a callback for `curlpp`, omits this parameter.
 *
 * @param ptr The location where data items are to be obtained from.
 * @param size The size in bytes of each single data item. For `curl` and
 *   `curlpp`, this size is always 1 (byte).
 * @param n_mem The number of data items.
 * @return The number of bytes successfully written to the target stream, which
 *   is handled by the consuming `curlpp` code. If this number of bytes does
 *   equal the total size of all data items, an error is supposed to be raised
 *   by the caller.
 */
size_t muted_write_function([[maybe_unused]] char *ptr, size_t size, size_t n_mem_b) {
    return size * n_mem_b;
}

/**
 * @brief Sets request headers for querying WikiData, meant for obtaining the
 *   labels of entities and properties.
 *
 * @param request The base request object to add request headers to.
 * @param ent_prp_part Entities and properties to obtain labels of.
 * @param language The natural language to obtain the labels in.
 */
void set_wikidata_request_headers(curlpp::Easy &request,
                                  const std::set<std::string> &ent_prp_part,
                                  const NaturalLanguage &language) {
    const std::string query = wikidata_labelling_query_for_entities_and_properties(ent_prp_part, language);
    const std::string encoded = url_encoded_string(query);
    request.setOpt(new curlpp::Options::HttpHeader({ "Accept: application/json",
                                                     "User-Agent: Curlpp/0.8.1" }));
    request.setOpt(new curlpp::Options::Url(wikidata_query_service_url +
                                            "sparql?query=" +
                                            encoded));
    request.setOpt(new curlpp::Options::WriteFunction(muted_write_function));
}

/**
 * @brief Returns the same entity-and-property labels JSON as what WikiData
 *   gave directly, except that it has been reduced to only the essential
 *   information: no XML data type information and other details.
 *
 * @param ent_prp_part An entity-and-property part of a partition.
 * @param unstructured The unstructured, raw labels JSON given by WikiData.
 * @return The cleaned-up JSON.
 */
Json::Value restructured_wikidata_entity_and_property_labels(const std::set<std::string> &ent_prp_part,
                                                             const Json::Value &unstructured) {
    Json::Value output;
    for (const auto &ent_or_prp : ent_prp_part) {
        /* Initially, each entity and property has an empty array of labels. */
        output[ent_or_prp] = Json::arrayValue;
    }
    for (const auto &binding : unstructured) {
        /* For each binding, which is essentially a single
         * entity-or-property-with-label pair, update the `output` map. */
        const std::string ent_or_prp = binding["id"]["value"].asString();
        const std::string label = binding["label"]["value"].asString();
        output[ent_or_prp].append(label);
    }
    return output;
}

const int too_many_requests_seconds_to_wait = 5;
const int query_interval_seconds_to_wait = 3;

/**
 * @brief Performs the WikiData query for obtaining entity and property labels,
 *   retrying when various types of network issues arise.
 *
 * @param request The WikiData label-extraction request to perform.
 */
void perform_wikidata_entity_and_property_labels_request(curlpp::Easy &request) {
    std::optional<size_t> res_code = std::nullopt;
    do {
        if (res_code.has_value() && res_code == 429) {
            std::this_thread::sleep_for(std::chrono::milliseconds(too_many_requests_seconds_to_wait * 1000));
        } else if (res_code.has_value()) {
            throw std::runtime_error(std::string("Received response code ") +
                                     std::to_string(res_code.value()) +
                                     " from WikiData. Aborting.");
        }
        request.perform();
        res_code = curlpp::Infos::ResponseCode::get(request);
    } while (res_code.has_value() && res_code.value() != 200);
    std::this_thread::sleep_for(std::chrono::milliseconds(query_interval_seconds_to_wait * 1000));
}

/**
 * @brief Returns the labels in `language` for the specified set of entities
 *   and properties, `ent_prp_part`.
 *
 * @param ent_prp_part An entity-and-property part of a partition.
 * @param language The natural language to get the labels in.
 * @return A mapping from entities and properties to arrays of zero or more
 *   labels in the required `language`.
 */
Json::Value entity_and_property_labels_of_part(const std::set<std::string> &ent_prp_part,
                                               const NaturalLanguage &language) {
    curlpp::Easy request;
    set_wikidata_request_headers(request, ent_prp_part, language);
    perform_wikidata_entity_and_property_labels_request(request);

    std::stringstream result;
    result << request;
    Json::Value json;
    result >> json;

    return restructured_wikidata_entity_and_property_labels(ent_prp_part, json["results"]["bindings"]);
}

/**
 * @brief Retrieves labels for WikiData entities and properties, serving a
 *   backend to `label_entities_and_properties`.
 *
 * @param split The LC-QuAD 2.0 dataset split to work on.
 * @param language The natural language to get labels for.
 * @param part_size The number of entities and properties to obtain labels for
 *   before saving to disk. The smaller, the more frequently is saved, but the
 *   slower the program runs.
 * @param quiet Whether to report on progress (`false`) or not (`true`).
 */
void label_entity_property_partitions_backend(const LCQuADSplit &split,
                                              const NaturalLanguage &language,
                                              int part_size,
                                              bool quiet) {
    std::set<std::string> require_labelling = entities_and_properties_requiring_labeling(split, language);
    ent_prp_partitioning partitioning = entity_property_partitioning(require_labelling, part_size);
    int count = 0;
    if (!quiet) {
        std::cout << "\rStarting with labelling entities and properties...";
        std::cout << std::flush;
    }
    for (const auto &part : partitioning) {
        Json::Value labels = entity_and_property_labels_of_part(part, language);
        save_entity_and_property_labels(labels, split, language);
        if (!quiet) {
            printf("\rRetrieved labels for part %5d/%5d (%6.2lf%%)",
                   count + 1,
                   static_cast<int>(partitioning.size()),
                   ((count + 1.) / static_cast<float>(partitioning.size())) * 100.);
            std::cout << std::flush;
        }
        count++;
    }
}

/**
 * @brief Collects labels for all WikiData entities and properties present in
 *   an LC-QuAD 2.0 dataset split.
 *
 * @param vm The variables map with which to determine how to approach the
 *   labelling operation. It determines which LC-QuAD 2.0 dataset split to
 *   collect entity-and-property labels for, in what natural language the
 *   labels should be expressed, how often to save to disk, and whether to
 *   report progress.
 */
void DutchKBQADSCreate::label_entities_and_properties(const po::variables_map &vm) {
    if (vm.count("split") == 0) {
        throw std::invalid_argument(std::string(R"(The "--split" flag )") +
                                    "is required.");
    } else if (vm.count("language") == 0) {
        throw std::invalid_argument(std::string(R"(The "--language" flag )") +
                                    "is required.");
    } else if (vm.count("part-size") == 0) {
        throw std::invalid_argument(std::string(R"(The "--part-size" flag )") +
                                    "is required.");
    } else if (vm.count("quiet") == 0) {
        throw std::invalid_argument(std::string(R"(The "--quiet" flag )") +
                                    "is required.");
    }
    const LCQuADSplit split = string_to_lc_quad_split_map.at(vm["split"].as<std::string>());
    const NaturalLanguage language = string_to_natural_language_map.at(vm["language"].as<std::string>());
    const int part_size = vm["part-size"].as<int>();
    const bool quiet = vm["quiet"].as<bool>();
    label_entity_property_partitions_backend(split, language, part_size, quiet);
}
