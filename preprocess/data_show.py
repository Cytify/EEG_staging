import data_load


def static_data(data):
    all_stage = {"W": 0, "1": 0, "2": 0, "3": 0, "4": 0, "R": 0}
    for file in data:
        curr_stage = {"W": 0, "1": 0, "2": 0, "3": 0, "4": 0, "R": 0}
        for seg in data[file]:
            all_stage[seg[1]] += 1
            curr_stage[seg[1]] += 1

        print("---------------- " + file + " situation ----------------")
        for stage in curr_stage:
            print(stage, "num", curr_stage[stage])
        print("\n")

    print("---------------- total situation ----------------")
    for stage in all_stage:
        print(stage, "num", all_stage[stage])


if __name__ == "__main__":
    static_data(data_load.load_data())
