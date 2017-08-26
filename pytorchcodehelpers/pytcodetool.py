import sys
import linecache

from pytorchcodehelpers.pytorchsize import InspectNet

if __name__ == '__main__':

    file_path = sys.argv[1]
    line_start = int(sys.argv[2])
    line_end = int(sys.argv[3])

    # file_path, line_start, line_end = ['/home/david/Documents/ws/ws/plygrnd/pytorchsize.py', 73, 109]

    lines = []

    for i in range(line_start, line_end + 1):
        lines.append(linecache.getline(file_path, i))

    lines = "".join(lines)

    inpt_size = input("Give the input size (default: 1 3 128 128) : ")
    inpt_size = inpt_size if inpt_size else "1 3 128 128"
    inpt_size = tuple(map(int, inpt_size.split(" ")))


    net_inspect = InspectNet(input_size=(inpt_size))
    net_inspect.inspect_net(lines)

    get_class_anwser = input("Get automatic generated class ? (default: N) : ")
    get_class_anwser = get_class_anwser if get_class_anwser else "N"

    if get_class_anwser.lower() == "y" or get_class_anwser.lower() == "yes":
        m_class_name = input("Class name ? (default: NewModule) : ")
        m_class_name = m_class_name if m_class_name else "NewModule"

        net_inspect.name = m_class_name

        print("\n")
        print(net_inspect.get_class_str())

    # print(lines)


    print("\nDone.")
