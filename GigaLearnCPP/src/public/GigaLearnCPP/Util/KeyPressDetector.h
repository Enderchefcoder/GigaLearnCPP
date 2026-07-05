#pragma once

#include "../Framework.h"

namespace GGL {
	namespace KeyPressDetector {
		// Returned by GetPressedChar() when input is unavailable (e.g. stdin is closed or not a terminal)
		constexpr char CHAR_UNAVAILABLE = '\0';

		// Blocks until a character is pressed and returns it
		// Returns CHAR_UNAVAILABLE if input is unavailable (do not keep polling in that case)
		char GetPressedChar();
	}
}
