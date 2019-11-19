import json
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--input', help='input file location')
parser.add_argument('--output', help='output file location containing the questions')

args=parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

with open(INPUT_FILE) as json_file:
    data = json.load(json_file)
correct_questions = data["correct_questions"]

with open (OUTPUT_FILE, "w", encoding="utf-8") as f:
    for group,questions in correct_questions.items():
        if len(questions) > 0:
            f.write(": {}\n".format(group))
            for question in questions:
                f.write("{} {} {} {}\n".format(",".join(question[0][0]),",".join(question[0][1]),",".join(question[1][0]),",".join(question[1][1])))
