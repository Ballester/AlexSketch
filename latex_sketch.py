import numpy as np

def latexFullSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1):
  latex = "\\begin{table}[tb]\n\
\caption{AlexNet + Full Sketch}\n\
\\resizebox{\columnwidth}{!}{%\n\
\\begin{tabular}{l c c c c}\n\
\hline\hline\n\
Classes & Correct 5 & Correct 1 & FP 5 & FP 1\\\\\n\
%heading\n\
\hline \\\n"

  for key in sorted(dict_expected):
    latex +=  key + ' & '
    latex += str(dict_correct_5[key]) if key in dict_correct_5 else '0'
    latex += ' & '
    latex += str(dict_correct_1[key]) if key in dict_correct_1 else '0'
    latex += ' & '
    latex += str(dict_false_positives_5[key]) if key in dict_false_positives_5 else '0'
    latex += ' & '
    latex += str(dict_false_positives_1[key]) if key in dict_false_positives_1 else '0'
    latex += ' \\\\\n'

  latex += '\hline %inserts single line\
\end{tabular}\n\
}\n\
\label{table:nonlin}\n\
\end{table}\
'

  return latex

def latexHalfSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1, trained_classes):
  latex = "\\begin{table}[tb]\n\
\caption{AlexNet + Half Sketch}\n\
\\resizebox{\columnwidth}{!}{%\n\
\\begin{tabular}{l c c c c c}\n\
\hline\hline\n\
Classes & Correct 5 & Correct 1 & FP 5 & FP 1 & Trained \\\\\n\
%heading\n\
\hline \\\n"

  for key in sorted(dict_expected):
    latex +=  key + ' & '
    latex += str(dict_correct_5[key]) if key in dict_correct_5 else '0'
    latex += ' & '
    latex += str(dict_correct_1[key]) if key in dict_correct_1 else '0'
    latex += ' & '
    latex += str(dict_false_positives_5[key]) if key in dict_false_positives_5 else '0'
    latex += ' & '
    latex += str(dict_false_positives_1[key]) if key in dict_false_positives_1 else '0'
    latex += ' & '
    latex += str(True) if key in trained_classes else str(False)
    latex += ' \\\\\n'

  latex += '\hline %inserts single line\\\\\n\
\end{tabular}\n\
}\n\
\label{table:nonlin}\n\
\end{table}\n\
'

  return latex

def latexNoSketch(dict_expected, dict_correct_5, dict_correct_1, dict_false_positives_5, dict_false_positives_1):
#   latex = "\\begin{table}[tb]\n\
# \caption{AlexNet}\n\
# \\resizebox{\columnwidth}{!}{%\n\
# \\begin{tabular}{l c c c c}\n\
# \hline\hline\n\
# Classes & Correct 5 & Correct 1 & FP 5 & FP 1\\\\\n\
# %heading\n\
# \hline\n"

#   for key in sorted(dict_expected):
#     latex +=  key + ' & '
#     latex += str(dict_correct_5[key]) if key in dict_correct_5 else '0'
#     latex += ' & '
#     latex += str(dict_correct_1[key]) if key in dict_correct_1 else '0'
#     latex += ' & '
#     latex += str(dict_false_positives_5[key]) if key in dict_false_positives_5 else '0'
#     latex += ' & '
#     latex += str(dict_false_positives_1[key]) if key in dict_false_positives_1 else '0'
#     latex += ' \\\\\n'

#   latex += '\hline %inserts single line\\\\\n\
# \end{tabular}\n\
# }\n\
# \label{table:nonlin}\n\
# \end{table}\
# '
  list_expected = []
  list_correct_5 = []
  list_correct_1 = []
  list_false_positives_5 = []
  list_false_positives_1 = []

  for key in sorted(dict_expected):
    list_expected.append(key)
    list_correct_5.append(dict_correct_5[key] if key in dict_correct_5 else '0')
    list_correct_1.append(dict_correct_1[key] if key in dict_correct_1 else '0')
    list_false_positives_5.append(dict_false_positives_5[key] if key in dict_false_positives_5 else '0')
    list_false_positives_1.append(dict_false_positives_1[key] if key in dict_false_positives_1 else '0')


  dicts = np.array(list_expected).reshape(len(list_expected), 1)
  list_correct_5 = np.array(list_correct_5).reshape(len(list_correct_5), 1)
  list_correct_1 = np.array(list_correct_1).reshape(len(list_correct_1), 1)
  list_false_positives_5 = np.array(list_false_positives_5).reshape(len(list_false_positives_5), 1)
  list_false_positives_1 = np.array(list_false_positives_1).reshape(len(list_false_positives_1), 1)

  dicts = np.concatenate((dicts, list_correct_5), axis=1)
  dicts = np.concatenate((dicts, list_correct_1), axis=1)
  dicts = np.concatenate((dicts, list_false_positives_5), axis=1)
  dicts = np.concatenate((dicts, list_false_positives_1), axis=1)

  return npToLatex(dicts)


def npToLatex(arr):
  latex = "\\begin{table}[tb]\n\
\caption{Table}\n\
\\resizebox{\columnwidth}{!}{%\n\
\\begin{tabular}{"
  #Find the number of columns, first one is an 'l':
  latex += "l"
  columns = arr.shape[1] - 1
  for i in range(0, columns):
    latex += " c"
  latex += "}\n\
\hline\hline\n\
Name "
  for i in range(0, columns):
    latex += "& Name"
  latex += "\\\\\n%heading\n\
\hline\n"

  #Insert values
  rows = arr.shape[0]
  for i in range(0, rows):
    latex += str(arr[i][0])
    for j in range(1, columns+1):
      latex += " & " + str(arr[i][j])
    latex += " \\\\\n"

  latex += '\hline %inserts single line\\\\\n\
\end{tabular}\n\
}\n\
\label{table:nonlin}\n\
\end{table}\
' 

  return latex
