# coding:utf-8


def replace_sep(fin, fout, sep_ini, sep_fin):
    """
    Replace delimiter in a file.
    """
    fin = open(fin, "r")
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace(sep_ini, sep_fin))
    fin.close()
    fout.close()


def remove_quotes(fin, fout):
    """ Remove quotes in lines.
    If a line has odd number quotes, remove all quotes in this line.
    """
    fin = open(fin)
    fout = open(fout, "w")
    for line in fin:
        fout.write(line.replace("\"", ""))
    fin.close()
    fout.close()
