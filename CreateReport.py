from MSTOkG.MSTOOpt import *
from joblib import Parallel, delayed
import sys
import time 
# from IPython.display import clear_output, display

from pylatex import Document, Section, Subsection, Figure, Command
from pylatex.utils import italic, NoEscape

dirname = "./testdata/"

doc = Document("MSTOKG_Status", documentclass='article', document_options=['a4paper,portrait'])

# Add a title and author to the preamble
doc.preamble.append(Command('title', 'MSTO-KG status'))
doc.preamble.append(Command('author', 'Subimal Deb'))
doc.preamble.append(Command('date', NoEscape(r'\today')))
doc.append(NoEscape(r'\maketitle'))


for element in 'H Li Be B C N O F'.split(' '):
    
    
    fig, T1, T2 = PrintEnergies(element, dirname = "./testdata/", table=True, plot=True, prefix="MOBS", 
                  save=True, kmax=11)#, ylim=[-0.5, -0.498])
    # doc.append(NoEscape(r'\tableofcontents'))
    print('\n'*2)
    doc.append(NoEscape(r'\newpage'))
    with doc.create(Section(element)):
        doc.append(NoEscape(T1.get_latex_string()))
        doc.append(NoEscape(r'\vskip 1cm '))
        doc.append(NoEscape(T2.get_latex_string()))
        # doc.append(NoEscape(r'\clearpage'))

    with doc.create(Figure(position='h!')) as figure:
        # Add the image
        # Ensure 'your_image.png' exists in the same directory or provide a full path
        figure.add_image(f'{element}.png', width='10cm') 
        # figure.add_caption('An example image in PyLaTeX.')
        figure.append(NoEscape(r'\label{fig:'+element+'}'))
        # doc.append(NoEscape(r'\newpage'))
        # doc.append(NoEscape(r'\clearpage'))



# Generate the .tex file and compile it to a PDF
# clean_tex=False keeps the generated .tex file for inspection
doc.generate_pdf(clean_tex=False)
# doc.generate_tex() # Generates only the .tex file