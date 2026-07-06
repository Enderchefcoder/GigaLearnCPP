#include "KeyPressDetector.h"

#ifdef _MSC_VER
#include <conio.h>

char GGL::KeyPressDetector::GetPressedChar() {
	int result = _getch();
	if (result == EOF)
		return CHAR_UNAVAILABLE;
	return (char)result;
}

#else
#include <unistd.h>
#include <termios.h>

char GGL::KeyPressDetector::GetPressedChar() {
	// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
	char buf = 0;

	// stdin may not be a terminal (e.g. piped input, headless server), in which case
	//	we just do a normal blocking read
	bool isTerminal = isatty(0);

	struct termios old = {};
	if (isTerminal) {
		if (tcgetattr(0, &old) < 0)
			perror("tcgetattr()");
		old.c_lflag &= ~ICANON;
		old.c_lflag &= ~ECHO;
		old.c_cc[VMIN] = 1;
		old.c_cc[VTIME] = 0;
		if (tcsetattr(0, TCSANOW, &old) < 0)
			perror("tcsetattr ICANON");
	}

	ssize_t readResult = read(0, &buf, 1);

	if (isTerminal) {
		old.c_lflag |= ICANON;
		old.c_lflag |= ECHO;
		if (tcsetattr(0, TCSADRAIN, &old) < 0)
			perror("tcsetattr ~ICANON");
	}

	if (readResult <= 0) {
		// 0 = end of input (stdin closed), <0 = read error
		// Either way, input is unavailable
		return CHAR_UNAVAILABLE;
	}

	return buf;
}

#endif
