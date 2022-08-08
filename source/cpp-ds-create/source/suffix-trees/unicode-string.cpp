/* Symbols for working with encoded Unicode strings. */

#include <algorithm>
#include "suffix-trees/unicode-string.hpp"

using namespace DutchKBQADSCreate::SuffixTrees;

/**
 * @brief Checks whether the UTF32-encoded Unicode string has a number of
 *   code points that is not excessive. Otherwise, it throws a
 *   `logic_error`.
 */
void UnicodeString::ensure_unicode_string_is_within_length_limit() {
    if (this->cp.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::logic_error(std::string("We currently only support ") +
                               "strings with a maximal size of " +
                               std::to_string(std::numeric_limits<int>::max()) +
                               " code points, inclusively!");
    }
}

/**
 * @brief Constructs a UTF32-encoded Unicode string from a regular C++ string.
 *
 * @param str The string to construct from.
 */
UnicodeString::UnicodeString(std::string str) {
    if (!utf8::is_valid(str.begin(), str.end())) {
        throw std::logic_error("String \"" + str + "\" is not properly UTF8-encoded!");
    }
    this->cp = {};
    utf8::iterator<std::string::const_iterator> it(str.begin(), str.begin(), str.end());
    utf8::iterator<std::string::const_iterator> end_it(str.end(), str.begin(), str.end());
    for (; it != end_it; ++it) {
        this->cp.push_back(*it);
    }
    this->ensure_unicode_string_is_within_length_limit();
    this->length = static_cast<int>(this->cp.size());
}

/**
 * @brief Constructs a UTF32-encoded Unicode string from a sequence of
 *   individual UTF32-encoded code points.
 *
 * @param code_points The code points to construct from.
 */
UnicodeString::UnicodeString(const std::vector<utf8::uint32_t> &code_points) {
    this->cp = { code_points };
    ensure_unicode_string_is_within_length_limit();
    this->length = static_cast<int>(this->cp.size());
}

/**
 * @brief Constructs a substring of the calling UTF32-encoded Unicode string.
 *
 * This method does not simply return a view into the already-existent 'parent'
 * Unicode string; it creates a wholly new UTF32-encoded string. Thus, it is
 * not very memory-efficient.
 *
 * @param start_index The starting index of the substring. Inclusive.
 * @param end_index The ending index of the substring. Exclusive.
 * @return The substring.
 */
UnicodeString UnicodeString::substring(int start_index, int end_index) {
    std::vector<utf8::uint32_t> substring_cp(this->cp.begin() + start_index,
                                             this->cp.begin() + end_index,
                                             this->cp.get_allocator());
    return UnicodeString(substring_cp);
}

/**
 * @brief Returns the UTF32-encoded code point at the requested `index`.
 *
 * @param index The index.
 * @return The code point.
 */
utf8::uint32_t UnicodeString::code_point_at(int index) {
    return this->cp[index];
}

/**
 * @brief Returns the index of the first occurrence of a UTF32-encoded Unicode
 *   code point, if it occurs in this Unicode string at all.
 *
 * @param code_point The code point to search for.
 * @return If at least one occurrence of `code_point` exists in this string,
 *   the index of the first occurrence of `code_point`. Otherwise, the C++ null
 *   value.
 */
std::optional<int> UnicodeString::index_of_code_point(utf8::uint32_t code_point) {
    auto it = std::find(this->cp.begin(), this->cp.end(), code_point);
    if (it != this->cp.end()) {
        return std::distance(this->cp.begin(), it);
    } else {
        return std::nullopt;
    }
}

/**
 * @brief Returns a basic string made out of C-style `char`s, based on a
 *   UTF32-encoded Unicode string.
 *
 * @param uni_str The UTF32-encoded Unicode string.
 * @return The basic string.
 */
std::basic_string<char> UnicodeString::basic_string_from_unicode_string(UnicodeString uni_str) {
    std::u32string utf_32_str(uni_str.cp.begin(), uni_str.cp.end(), uni_str.cp.get_allocator());
    return { utf8::utf32to8(utf_32_str) };
}

/**
 * @brief Returns a basic string made out of C-style `char`s, based on a single
 *   UTF32-encoded Unicode code point.
 *
 * @param code_point The UTF32-encoded code point.
 * @return The basic string.
 */
std::basic_string<char> UnicodeString::basic_string_from_unicode_code_point(utf8::uint32_t code_point) {
    std::u32string utf_32_str({ code_point });
    return { utf8::utf32to8(utf_32_str) };
}

/**
 * @brief Returns the unique UTF32-encoded Unicode code points that can be
 *   found in the calling UTF32 Unicode string.
 *
 * @return The unique code points.
 */
std::set<utf8::uint32_t> UnicodeString::unique_code_points() {
    std::set<utf8::uint32_t> code_points{};
    auto add_to_code_points = [&code_points] (const utf8::uint32_t &code_point) -> void {
        code_points.insert(code_point);
    };
    std::for_each(this->cp.begin(), this->cp.end(), add_to_code_points);
    return code_points;
}
