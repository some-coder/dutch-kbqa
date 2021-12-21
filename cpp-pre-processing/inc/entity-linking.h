#ifndef ENTITY_LINKING_H
#define ENTITY_LINKING_H


#include <map>
#include <vector>


typedef enum Language {
	ENGLISH,
	DUTCH
} Language;


typedef enum Mode {
	TRAIN,
	TEST
} Mode;


/**
 * Yields a mapping from UIDs to questions in natural language.
 *
 * @param lang The language the sentences should have.
 * @param mode The mode of the data being read in: training or testing.
 * @return The mapping.
 */
std::map<int, std::string> sentences_with_u_ids(Language lang, Mode mode);

/**
 * Yields a mapping from UIDs to WikiData Q- and P-values (entities and relations, respectively).
 *
 * @param mode The mode of the Q- and P-values.
 * @return The mapping.
 */
std::map<int, std::vector<std::string>> uid_q_p_values(Mode mode);

/**
 * Obtains the lexemes associated with all known Q- and P-values in the dataset.
 *
 * @param lang The language of the dataset.
 * @param mode The mode for which the dataset is targeted for use.
 * @return The mapping.
 */
std::map<std::string, std::vector<std::string>> q_p_value_lexemes(Language lang, Mode mode);

/**
 * Determines the longest common substrings (LCSs) of a set of lexeme of a Q- or P-value, evaluated on a sentence.
 *
 * @param sen The sentence to compute the LCSs for.
 * @param lexemes The lexemes to derive LCSs for.
 * @return The LCSs.
 */
std::map<std::string, std::string> q_p_value_longest_common_substrings(
		const std::string& sen,
		const std::vector<std::string>& lexemes
	);

/**
 * Maps UIDs to WikiData Q- and P-values, which map to lexemes, which in turn map to LCSs.
 *
 * @param sen_u_ids The sentences with associated UIDs.
 * @param uid_q_ps The UIDs with associated WikiData Q- and P-values.
 * @param lang The language of the sentences.
 * @param mode The mode of the dataset from which the sentences were derived.
 * @return The mapping.
 */
std::map<int, std::map<std::string, std::map<std::string, std::string>>>
	uid_longest_common_substrings(
		const std::map<int, std::string>& sen_u_ids,
		std::map<int, std::vector<std::string>> uid_q_ps,
		Language lang,
		Mode mode
	);

/**
 * Saves the sentences' LCSs to disk as a JSON file.
 *
 * @param uid_lcs The mapping to save.
 * @param lang The language of the data that was read in.
 * @param mode The mode the data that was read in.
 */
void save_to_disk(
	const std::map<int, std::map<std::string, std::map<std::string, std::string>>>& uid_lcs,
	Language lang,
	Mode mode);


#endif /* ENTITY_LINKING_H */
