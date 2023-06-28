def write_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)




if __name__ == '__main__':
    file_path = "record.txt"
    content = "hello\n"
    write_to_file(file_path, content)
    print("@@@@ main func finish")
