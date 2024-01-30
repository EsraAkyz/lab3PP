#include "cuda/image.cuh"

#include "encryption/Algorithm.h"
#include "encryption/FES.h"
#include "encryption/Key.h"
#include "io/image_parser.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <bit>

int main(int argc, char** argv) {
	auto lab_cli_app = CLI::App{ "" };

	auto image_path = std::filesystem::path{};
	auto file_option = lab_cli_app.add_option("--file", image_path);

	auto output_path = std::filesystem::path{};
	auto output_option = lab_cli_app.add_option("--output", output_path);

	file_option->check(CLI::ExistingFile);
	output_option->check(CLI::ExistingDirectory);

	CLI11_PARSE(lab_cli_app, argc, argv);

	auto loaded_image = ImageParser::read_bitmap(image_path);

	auto key_1 = Key::get_standard_key();

	auto img_16 = loaded_image;
	auto img_15 = img_16;
	auto img_14 = img_15;
	auto img_13 = img_14;
	auto img_12 = img_13;
	auto img_11 = img_12;
	auto img_10 = img_11;
	auto img_09 = img_10;
	auto img_08 = img_09;
	auto img_07 = img_08;
	auto img_06 = img_07;
	auto img_05 = img_06;
	auto img_04 = img_05;
	auto img_03 = img_04;
	auto img_02 = img_03;
	auto img_01 = img_02;

	ImageParser::write_bitmap(output_path / "decrypted.bmp", img_01);

	return 0;
}
