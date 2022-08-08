/* Symbols for working with encoded Unicode strings (header). */

#ifndef UNICODE_STRING_HPP
#define UNICODE_STRING_HPP

#include <vector>
#include <set>
#include <optional>
#include "utf8.h"

namespace DutchKBQADSCreate::SuffixTrees {
    /**
     * @brief A convenience class for working with UTF32-encoded Unicode
     *   strings.
     *
     * This class is by no means meant to be performant.
     */
    class UnicodeString {
    private:
        std::vector<utf8::uint32_t> cp;  /* The UTF32-encoded code points. */
        void ensure_unicode_string_is_within_length_limit();
    public:
        int length;
        explicit UnicodeString(std::string str);
        explicit UnicodeString(const std::vector<utf8::uint32_t>& code_points);
        UnicodeString substring(int start_index, int end_index);
        utf8::uint32_t code_point_at(int index);
        std::optional<int> index_of_code_point(utf8::uint32_t code_point);
        static std::basic_string<char> basic_string_from_unicode_string(UnicodeString uni_str);
        static std::basic_string<char> basic_string_from_unicode_code_point(utf8::uint32_t code_point);
        std::set<utf8::uint32_t> unique_code_points();
    };
}

#endif  /* UNICODE_STRING_HPP */