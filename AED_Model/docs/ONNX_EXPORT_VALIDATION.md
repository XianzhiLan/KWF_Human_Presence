{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue-Bold;\f1\fnil\fcharset0 HelveticaNeue;\f2\fnil\fcharset0 Menlo-Regular;
}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\sa40\partightenfactor0

\f0\b\fs32 \cf0 ONNX Export Validation\
\pard\pardeftab560\slleading20\partightenfactor0

\f1\b0\fs26 \cf0 The TinyCNN V3 model was successfully exported to the ONNX format.\
To validate the export, inference was run using both the original PyTorch model and the exported ONNX model on the same dataset. One file, 
\f0\b Audio_Moth_5_20250318_043905.wav
\f1\b0 , was corrupted and could not be processed, so it was skipped during inference.\

\f0\b Results:
\f1\b0 \
\pard\pardeftab560\pardirnatural\partightenfactor0
\ls1\ilvl0
\f2\fs18 \cf0 {\listtext	\uc0\u8226 	}
\f1\fs26 ONNX export: 
\f0\b Successful
\f1\b0 \
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f1\fs26 Clips compared: 
\f0\b 631,316
\f1\b0 \
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f1\fs26 Corrupted file skipped: 
\f0\b Audio_Moth_5_20250318_043905.wav
\f1\b0 \
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f1\fs26 Maximum absolute probability difference: 
\f0\b 1.788 \'d7 10\uc0\u8315 \u8310 
\f1\b0 \
\ls1\ilvl0
\f2\fs18 {\listtext	\uc0\u8226 	}
\f1\fs26 Mean absolute probability difference: 
\f0\b 3.496 \'d7 10\uc0\u8315 \u8313 
\f1\b0 \
\pard\pardeftab560\slleading20\partightenfactor0
\cf0 All processed clips matched within the tolerance of 
\f0\b 0.0001
\f1\b0 , indicating that the exported ONNX model produces effectively identical outputs to the original PyTorch model.\
}