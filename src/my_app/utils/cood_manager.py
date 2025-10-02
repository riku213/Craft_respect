class CoodManager:
    def get_cood_list(self,images_paths):
        ret_list = []
        for image_path in images_paths:
            path_list = str(image_path).split('\\')
            doc_id = path_list[-3]
            image_id = path_list[-1].replace('.jpg', '')
            ret_list.append(
                self.get_one_cood_list(
                    doc_id=doc_id,
                    image_id=image_id
                    )
                )
        return ret_list

    def get_one_cood_list(self, doc_id, image_id):
        csv_path = f'../../kuzushiji-recognition/char_sep_datas/{doc_id}/{doc_id}_coordinate.csv'
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readline()
            ret_list = []
            while True:
                line = f.readline().rstrip()
                if not line:
                    break
                line_list = line.split(',')
                # print(line_list[1], image_id) 
                if line_list[1] == image_id:
                    # print('found!')
                    cood_list = [
                        int(line_list[2]),
                        int(line_list[3]),
                        int(line_list[6]),
                        int(line_list[7])
                    ]
                    ret_list.append(cood_list)
                else:
                    continue
        return ret_list