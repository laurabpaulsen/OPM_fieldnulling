import struct
import sys
from utils.compcoils import CompFieldControl
import argparse as ap

def intBitsToFloat(b):
   s = struct.pack('>l', b)
   return struct.unpack('>f', s)[0]

array = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '30635', '30635', '30635', '553678763', '30635', '30635', '30635', '30635', '50362283', '30635', '30635', '30635', '587233195', '604010411', '553678763', '33583239', '570455979', '553678763', '570455979', '30635', '30635', '30635', '553678763', '570455979', '30635', '30635', '33585067', '587233195', '30635', '587233195', '30635', '33583111', '30635', '50362283', '30635', '620787627', '30635', '30635', 
'30635', '30635', '30635', '620787627', '620787627', '50362283', '570455979', '30635', '553678763', '30635']

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--which_coils', type=str, default='0', help='Input the index of the coil to change or "all" and input a "--list"')
    parser.add_argument('--field', type=float, default=0.0, help='Which field strength to apply to the chosen coil channel')
    parser.add_argument('--list', type=float, nargs="*", default=[0,0,0,0,0,0,0,0], help='If which_coils is "all" supply a list if with coil settings')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
#    which_num = int(input("Give index for Uint32_t array!"))
    # print(len(array))
#    packed_v = struct.pack('>l', b)
    # print(intBitsToFloat(30635))
    # val = '1803'
    # # val = int(1803)
    # print(val)
    # # print(str(val))
    # # print(f"0x{val:08X}")
    # print(str(format(int(val), "032b"))[13:])

# 0000-0000 0000-0000 0111-0111 1010-1011
# 0000000000000 0000111011100101011
# 0000000000000 0000000011100001011
    coilcontrol = CompFieldControl()
    args = parse_args()

    if args.which_coils == "all" and isinstance(args.list,list):
        set_field = list(args.list)
        if len(set_field) != 8:
            # set_field = [5.527e+01,  8.756e+01, -9.785e-01, -1.715e+00, -3.215e-01, 3.833e+00,  7.529e-01,  2.291e+00]
            set_field = [0,0,0,0,0,0,0,0]
            # print('List given is too short') 
        # print(set_field)
        coilcontrol.set_coil_values(set_field)


    elif int(args.which_coils) in [0,1,2,3,4,5,6,7]:
        set_field = args.field
        if isinstance(set_field,float):
            # print([int(args.which_coils),set_field])
            coilcontrol.setOffset(int(args.which_coils),set_field)
        else:
            print('not a float?')

    else:
        print('fuck!!')

    coilcontrol.ser_monitor.join()
 