import argparse


def parse_range(years_str: str) -> list[int]:
	parts = years_str.split(",")
	result = []
	for part in parts:
		sub_parts = part.split("-")
		if len(sub_parts) == 1:
			result.append(int(sub_parts[0]))
		elif len(sub_parts) == 2:
			result.extend(list(range(int(sub_parts[0]), int(sub_parts[1]) + 1)))
		else:
			raise argparse.ArgumentTypeError("Invalid series format")
	return result
