#include "Utils.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <unistd.h>
#endif

std::filesystem::path GGL::Utils::GetExecutableDir() {
#ifdef _WIN32
	wchar_t buffer[MAX_PATH] = {};
	DWORD size = GetModuleFileNameW(NULL, buffer, MAX_PATH);
	if (size > 0 && size < MAX_PATH)
		return std::filesystem::path(buffer).parent_path();
#else
	char buffer[4096] = {};
	ssize_t size = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
	if (size > 0)
		return std::filesystem::path(std::string(buffer, size)).parent_path();
#endif

	return std::filesystem::current_path();
}

std::set<int64_t> GGL::Utils::FindNumberedDirs(std::filesystem::path basePath) {
	std::set<int64_t> results = {};

	if (!std::filesystem::exists(basePath))
		return results;

	for (auto entry : std::filesystem::directory_iterator(basePath)) {
		if (entry.is_directory()) {
			auto name = entry.path().filename();
			bool isNameAllNumbers = true;
			for (char c : name.string()) {
				if (!isdigit(c)) {
					isNameAllNumbers = false;
					break;
				}
			}
			if (!isNameAllNumbers)
				continue;

			results.insert(std::stoll(name.string()));
		}
	}

	return results;
}