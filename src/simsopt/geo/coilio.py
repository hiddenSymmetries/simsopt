# coilio.py: methods to extract geometry from different coil conventions. 
# such as MAKEGRID (x,y,z,current,coilgroup)
# and IPP. 


__all__ = ['get_data_from_makegrid', 'get_data_from_xyzfile']


def  get_data_from_xyzfile(filename):
    """
    get the x,y,z coordinates from a file that contains 
    some header lines and then lines with x, y, and z locations. 
    This convention is in use at IPP, 
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()  # skip the first 3 header lines
        splitlines = [line.split() for line in lines] 
    
    single_curve_data = []  # running storage for a single coil
    total_lines = splitlines[0][0]  # first line contains the number of lines
    
    for line in splitlines:
        if len(line) == 3:
            single_curve_data.append([float(x) for x in line])
    
    if len(single_curve_data) != int(total_lines):
        raise ValueError(f"File {filename} does not contain the expected number of lines. Expected {total_lines}, found {len(single_curve_data)}")

    return single_curve_data


def get_data_from_makegrid(filename, group_names=None):
    """
    get the x,y,z coordinates from a a MAKEGRID type file. 

    These files contain 3 header lines defining the contents of the
    file, followed by lines with coil information. 
    This is structured as follows: 
    x, y, z, current, [group_number], [group_name]
    Where group_number and group_name only occur on the
    line where a new coil starts. 
    Here we use line length to determine the last coordinates
    to append to a coil. 
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()[3:]  # skip the first 3 header lines
        splitlines = [line.split() for line in lines] 
    
    curve_data = []  # final list of lists
    single_curve_data = []  # running storage for a single coil
    
    for linenum, line in enumerate(splitlines):
        if len(line) ==4:  #line contains x,y,z current
            single_curve_data.append([float(x) for x in line[:3]])
        elif len(line) == 6:  # line contains x,y,z current, group_number
            single_curve_data.append([float(x) for x in line[:3]])
            this_group_name = line[5]
            if group_names is None or this_group_name in group_names:  # we want this group
                curve_data.append(single_curve_data)
            single_curve_data = []
        elif len(line) == 1:  # MGRID sometimes ends with 'end'
            break
        else:
            raise ValueError(f"File does not adhere to makegrid conventions on line number {linenum+3}: {line}. Should contain 4 or 6 elements or read 'end'")
    
    return curve_data





