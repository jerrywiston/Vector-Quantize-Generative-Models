#########################
# Load .obj file
#########################
def load(path):
    v = []
    vn = []
    f = []
    vt = []
    file = open(path)
    while True:
        content = file.readline()
        if content == "":
            return v, vn, f, vt
        temp = content.strip().split(" ")
        if temp[0] == "v":
            v.append([float(temp[1]), float(temp[2]), float(temp[3])])
        elif temp[0] == "vn":
            vn.append([float(temp[1]), float(temp[2]), float(temp[3])])
        elif temp[0] == "vt":
            vt.append([float(temp[1]), float(temp[2])])
        elif temp[0] == "f":
            f1 = int(temp[1].split("/")[0])
            f2 = int(temp[2].split("/")[0])
            f3 = int(temp[3].split("/")[0])
            f.append([f1, f2, f3])

def load_(path, v_offset=(0.,0.,0.), f_offset=0, scale=0.5, tex_num=1, tex_id=0):
    v = []
    vn = []
    f = []
    vt = []
    file = open(path)
    while True:
        content = file.readline()
        if content == "":
            return v, vn, f, vt, len(v)
        temp = content.strip().split(" ")
        if temp[0] == "v":
            v.append([scale*float(temp[1])+v_offset[0], scale*float(temp[2])+v_offset[1], scale*float(temp[3])+scale*v_offset[2]])
        elif temp[0] == "vn":
            vn.append([float(temp[1]), float(temp[2]), float(temp[3])])
        elif temp[0] == "vt":
            vt1 = (1 + float(temp[1]) + tex_id*3) / (3*tex_num)
            vt2 = float(temp[2])
            vt.append([vt1, vt2])
        elif temp[0] == "f":
            f1 = int(temp[1].split("/")[0]) + int(f_offset) -1
            f2 = int(temp[2].split("/")[0]) + int(f_offset) -1
            f3 = int(temp[3].split("/")[0]) + int(f_offset) -1
            f.append([f1, f2, f3])

if __name__ == "__main__":
    v, vn, f, vt = load(path = './resource/obj/Cube.obj')
    print("v", v)   
    print("vn", vn)
    print("f", f)
    print("vt", vt)