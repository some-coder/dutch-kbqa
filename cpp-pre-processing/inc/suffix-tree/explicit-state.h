#ifndef EXPLICIT_STATE_H
#define EXPLICIT_STATE_H


#include "utf8.h"

#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <variant>

#include "unicode-string.h"


class ExplicitState;


typedef std::variant<std::unique_ptr<int>, int*> RightPointer;


class ExplicitState {

	static int ID_COUNTER;

	private:
		const int id;
		ExplicitState *parent;
		std::map<utf8::uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>> transitions;
		ExplicitState *suffix_link;

	public:
		explicit ExplicitState(ExplicitState *parent);
		virtual ~ExplicitState();
		void set_transition(
			UnicodeString uni_str,
			int left_ptr,
			RightPointer right_ptr,
			std::unique_ptr<ExplicitState> child);
		void set_suffix_link(ExplicitState *next_on_path);
		ExplicitState *get_suffix_link();
		virtual bool has_transition(utf8::uint32_t letter);
		virtual std::pair<std::pair<int, int*>, ExplicitState*> weakly_get_transition(utf8::uint32_t letter);
		ExplicitState *internal_split(
			UnicodeString uni_str,
			int left_ptr,
			RightPointer right_ptr);
		virtual std::optional<ExplicitState*> t_transition(utf8::uint32_t letter);
		std::map<uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>::iterator
			transitions_start();
		std::map<uint32_t, std::pair<std::pair<int, RightPointer>, std::unique_ptr<ExplicitState>>>::iterator
			transitions_end();
		friend std::ostream& operator<<(std::ostream& os, const ExplicitState& es);
		virtual void print(UnicodeString uni_str, int num_indents);

};


class AuxiliaryState : public ExplicitState {

	private:
		std::unique_ptr<ExplicitState> root;
		std::map<utf8::uint32_t, std::unique_ptr<int>> j_map;  /* Page 257, algorithm 2, line 2 of Ukkonen (1995). */

	public:
		explicit AuxiliaryState(UnicodeString uni_str);
		~AuxiliaryState() override;
		bool has_transition(utf8::uint32_t letter) override;
		std::pair<std::pair<int, int*>, ExplicitState*> weakly_get_transition(utf8::uint32_t letter) override;
		std::optional<ExplicitState *> t_transition(utf8::uint32_t letter) override;
		void print(UnicodeString uni_str, int num_indents) override;

};


#endif  /* EXPLICIT_STATE_H */
