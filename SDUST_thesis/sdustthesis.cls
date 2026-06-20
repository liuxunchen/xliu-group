%%
%% 山东科技大学学位论文模板 SDUSTThesis (v3.1 修复版)
%% 英文主字体：TeX Gyre Termes（Times 风格）
%% 中文字体：Fandol（宋体风格）
%% 修复内容：
%%   1. 移除对 \normalfont 的危险重定义，解决正文格式从标题泄露问题
%%   2. 优化 chapter afterskip，解决标题下方空间不足问题
%%   3. 修复 \chaptereng 分组缺失导致的居中泄露
%%   4. 统一行距控制，移除 \mainmatter 中的硬编码 baselineskip
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdustthesis}[2026/05/25 v3.1 SDUST Thesis Template]

% ==================== 选项处理 ====================
\newif\ifsdust@review\sdust@reviewfalse
\newif\ifsdust@twoside\sdust@twosidetrue
\newif\ifsdust@openright\sdust@openrighttrue
\newif\ifsdust@bachelor\sdust@bachelorfalse
\newif\ifsdust@master\sdust@masterfalse
\newif\ifsdust@doctor\sdust@doctorfalse

\DeclareOption{review}{\sdust@reviewtrue}
\DeclareOption{oneside}{\sdust@twosidefalse}
\DeclareOption{twoside}{\sdust@twosidetrue}
\DeclareOption{openany}{\sdust@openrightfalse}
\DeclareOption{openright}{\sdust@openrighttrue}
\DeclareOption{bachelor}{\sdust@bachelortrue\sdust@masterfalse\sdust@doctorfalse}
\DeclareOption{master}{\sdust@mastertrue\sdust@bachelorfalse\sdust@doctorfalse}
\DeclareOption{doctor}{\sdust@doctortrue\sdust@masterfalse\sdust@bachelorfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessOptions\relax

% ==================== 基础类加载 ====================
\LoadClass[zihao=-4, linespread=1.2, fontset=fandol,
  \ifsdust@twoside twoside\else oneside\fi,
  \ifsdust@openright openright\else openany\fi]{ctexbook}

\RequirePackage{fontspec}
\RequirePackage{etoolbox}

% ==================== 字体设置 ====================
\setmainfont{TeX Gyre Termes}[Ligatures=TeX]
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}

% ==================== 页面设置 ====================
\RequirePackage{geometry}
\geometry{
  a4paper,
  top=25mm,
  bottom=20mm,
  inner=25mm,
  outer=20mm,
  headheight=8mm,
  headsep=5mm,
  footskip=10mm,
  includehead,
  includefoot
}

% ==================== 正文默认格式（安全版）====================
% 不再重定义 \normalfont，避免格式泄露和宏包冲突
\AtBeginDocument{%
  \setlength{\parskip}{0pt}%
  \setlength{\parindent}{2em}%
  \songti
  \zihao{-4}%
}

% ==================== 目录名称 ====================
\renewcommand{\listfigurename}{图清单}
\renewcommand{\listtablename}{表清单}
\renewcommand{\contentsname}{目\,录}

% ==================== 页眉页脚 ====================
\RequirePackage{fancyhdr}
\fancypagestyle{sdust@headings}{%
  \fancyhf{}%
  \fancyhead[L]{\zihao{5}\songti 山东科技大学硕士学位论文}%
  \fancyhead[R]{\zihao{5}\songti \thepage \quad \leftmark}%
  \renewcommand{\headrulewidth}{0.4pt}%
}
\pagestyle{sdust@headings}
\renewcommand{\chaptermark}[1]{\markboth{#1}{#1}}

% ==================== 章、节格式（修复间距与泄露）====================
\ctexset{
  chapter = {
    format      = {\centering\zihao{-2}\heiti\bfseries},
    name        = {},
    number      = \chinese{chapter},
    beforeskip  = 24pt,
    afterskip   = 24pt,   % 修复：从 0pt 增至 24pt，解决标题下方空间不足
    fixskip     = true,
  },
  section = {
    format      = {\zihao{3}\heiti\bfseries},
    beforeskip  = 18pt,
    afterskip   = 12pt,
    fixskip     = true,
  },
  subsection = {
    format      = {\zihao{4}\heiti\bfseries},
    beforeskip  = 12pt,
    afterskip   = 12pt,
    fixskip     = true,
  },
  subsubsection = {
    format      = {\zihao{-4}\normalfont},
    beforeskip  = 12pt,
    afterskip   = 12pt,
    fixskip     = true,
  },
}

% 修复：用分组包裹防止 \centering 泄露到正文
\newcommand{\chaptereng}[1]{%
  {\par\vspace{0pt}\centering\zihao{-2}\bfseries\MakeUppercase{#1}\par}%
  \vspace{18pt plus 2pt minus 1pt}%
}

% ==================== 目录格式 ====================
\RequirePackage{titletoc}
\titlecontents{chapter}[0pt]
  {\addvspace{12pt}\zihao{4}\heiti\bfseries}
  {\contentspush{\thecontentslabel\quad}}{}
  {\titlerule*[5pt]{.}\contentspage}
\titlecontents{section}[1em]
  {\zihao{-4}}
  {\contentspush{\thecontentslabel\quad}}{}
  {\titlerule*[5pt]{.}\contentspage}
\titlecontents{subsection}[2em]
  {\zihao{5}}
  {\contentspush{\thecontentslabel\quad}}{}
  {\titlerule*[5pt]{.}\contentspage}

% ==================== 图表标题 ====================
\RequirePackage{caption}
\captionsetup{labelsep=quad, font={small, rm}, skip=5bp, justification=centering}
\captionsetup[figure]{position=bottom, name=图}
\captionsetup[table]{position=top, name=表}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}
\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}

% ==================== 公式编号 ====================
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}
\@addtoreset{equation}{chapter}

% ==================== 脚注 ====================
\RequirePackage[perpage, bottom]{footmisc}
\renewcommand{\footnotesize}{\zihao{5}}
\setlength{\footnotesep}{10bp}

% ==================== 常用宏包 ====================
\RequirePackage{amsmath, amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs, multirow, longtable}
\RequirePackage{listings}
\RequirePackage{hologo}
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\RequirePackage{float}
\RequirePackage{enumitem}
\setenumerate[1]{itemsep=0pt, parsep=0pt, topsep=5pt}
\setitemize[1]{itemsep=5pt, parsep=5pt, topsep=5pt}

% ==================== 用户信息存储 ====================
\newcommand{\sdust@infostore}[2]{\@namedef{sdust@#1}{#2}}
\newcommand{\sdust@infoget}[1]{\@nameuse{sdust@#1}}

\newcommand{\sdustsetup}[1]{\setkeys{sdust}{#1}}
\RequirePackage{kvsetkeys}
\define@key{sdust}{zh/title}{\sdust@infostore{zh/title}{#1}}
\define@key{sdust}{en/title}{\sdust@infostore{en/title}{#1}}
\define@key{sdust}{zh/author}{\sdust@infostore{zh/author}{#1}}
\define@key{sdust}{en/author}{\sdust@infostore{en/author}{#1}}
\define@key{sdust}{zh/supervisor}{\sdust@infostore{zh/supervisor}{#1}}
\define@key{sdust}{en/supervisor}{\sdust@infostore{en/supervisor}{#1}}
\define@key{sdust}{zh/major}{\sdust@infostore{zh/major}{#1}}
\define@key{sdust}{en/major}{\sdust@infostore{en/major}{#1}}
\define@key{sdust}{zh/department}{\sdust@infostore{zh/department}{#1}}
\define@key{sdust}{zh/keywords}{\sdust@infostore{zh/keywords}{#1}}
\define@key{sdust}{en/keywords}{\sdust@infostore{en/keywords}{#1}}
\define@key{sdust}{id}{\sdust@infostore{id}{#1}}
\define@key{sdust}{enrollment}{\sdust@infostore{enrollment}{#1}}
\define@key{sdust}{defense}{\sdust@infostore{defense}{#1}}
\define@key{sdust}{secretlevel}{\sdust@infostore{secretlevel}{#1}}
\define@key{sdust}{ccode}{\sdust@infostore{ccode}{#1}}
\define@key{sdust}{udc}{\sdust@infostore{udc}{#1}}
\define@key{sdust}{degreetype}{\sdust@infostore{degreetype}{#1}}
\define@key{sdust}{degreelevel}{\sdust@infostore{degreelevel}{#1}}
\define@key{sdust}{researchfield}{\sdust@infostore{researchfield}{#1}}
\define@key{sdust}{eduduration}{\sdust@infostore{eduduration}{#1}}
\define@key{sdust}{graduyear}{\sdust@infostore{graduyear}{#1}}
\define@key{sdust}{submitdate}{\sdust@infostore{submitdate}{#1}}
\define@key{sdust}{supertitle}{\sdust@infostore{supertitle}{#1}}
\define@key{sdust}{reviewers}{\sdust@infostore{reviewers}{#1}}
\define@key{sdust}{defensechair}{\sdust@infostore{defensechair}{#1}}
\define@key{sdust}{defensecommittee}{\sdust@infostore{defensecommittee}{#1}}
\define@key{sdust}{totalpages}{\sdust@infostore{totalpages}{#1}}
\define@key{sdust}{fund}{\sdust@infostore{fund}{#1}}
\define@key{sdust}{zh/display-date}{\sdust@infostore{zh/display-date}{#1}}
\define@key{sdust}{en/display-date}{\sdust@infostore{en/display-date}{#1}}

% 默认值
\sdustsetup{
  zh/title = {山东科技大学学位论文},
  en/title = {SDUST Thesis},
  zh/author = {姓名},
  zh/supervisor = {导师},
  zh/major = {专业},
  id = {2023000001},
  enrollment = {2023年9月},
  defense = {2026年6月},
  secretlevel = {公开},
  ccode = {TB532},
  udc = {621},
  degreetype = {专业学位},
  degreelevel = {硕士},
  researchfield = {研究方向},
  eduduration = {三年},
  graduyear = {2026},
  submitdate = {2026年6月},
  supertitle = {教授},
  reviewers = {匿名评审},
  defensechair = {待定},
  defensecommittee = {待定},
  totalpages = {100},
  fund = {},
  zh/display-date = {\the\year 年\the\month 月},
}

% ==================== 封面 ====================
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \vspace*{20mm}
    \includegraphics[width=0.3\textwidth]{sdustlogo.png}\\
    \vspace{10mm}
    \ifsdust@doctor
      {\zihao{-2}\kaishu\bfseries 博士学位论文\par}
    \else
      \ifsdust@master
        {\zihao{-2}\kaishu\bfseries 硕士学位论文\par}
      \else
        {\zihao{-2}\kaishu\bfseries 学士学位论文\par}
      \fi
    \fi
    \vspace{15mm}
    {\zihao{4}\heiti\bfseries\sdust@infoget{zh/title}\par}
    \vspace{12mm}
    {\zihao{-2}\bfseries\selectfont\MakeUppercase{\sdust@infoget{en/title}}\par}
    \vspace{15mm}
    {\zihao{3}\songti
      \begin{tabular}{rl}
        作 者： & \sdust@infoget{zh/author} \\[6pt]
        导 师： & \sdust@infoget{zh/supervisor} \\[6pt]
      \end{tabular}
    }
    \vfill
    {\zihao{-2}\kaishu 山东科技大学}\\[6pt]
    {\zihao{-2}\kaishu\sdust@infoget{zh/display-date}}
    \vspace{10mm}
  \end{titlepage}
}

% ==================== 声明页 ====================
\newcommand{\copyrightpage}{%
  \cleardoublepage
  \begin{center}
    {\zihao{-2}\heiti\bfseries 学位论文使用授权声明}
  \end{center}
  \vspace{12pt}
  \par\zihao{-4}\kaishu
  本人完全了解山东科技大学有关保留、使用学位论文的规定，同意本人所撰写的学位论文的使用授权按照学校的管理规定处理。\par
  作为申请学位的条件之一，学校有权保留学位论文并向国家有关部门或其指定机构送交论文的电子版和纸质版；有权将学位论文的全部或部分内容编入有关数据库发表，并可以以电子、网络及其他数字媒体形式公开出版；允许学校档案馆和图书馆保留学位论文的纸质版和电子版，可以使用影印、缩印或扫描等复制手段保存和汇编学位论文；为教学和科研目的，学校档案馆和图书馆可以将公开的学位论文作为资料在档案馆、图书馆等场所或在校园网上供校内师生阅读、浏览。\par
  \vspace{15mm}
  \hfill 作者签名：\hspace{3cm} 导师签名：\hspace{3cm}\par
  \hfill 日期：\hspace{2cm}年\hspace{1cm}月\hspace{1cm}日\hspace{2cm} 日期：\hspace{2cm}年\hspace{1cm}月\hspace{1cm}日\par

  \cleardoublepage
  \begin{center}
    {\zihao{-2}\heiti\bfseries 学位论文原创性声明}
  \end{center}
  \vspace{12pt}
  \par\zihao{-4}\kaishu
  本人呈交给山东科技大学的学位论文，除所列参考文献和世所公认的文献外，全部是本人攻读学位期间在导师指导下的研究成果。除文中已经标明引用的内容外，本论文不包含任何其他个人或集体已经发表或撰写过的研究成果。对本文的研究做出贡献的个人和集体，均已在文中以明确方式标明。本人完全意识到本声明的法律结果由本人承担。\par
  若有不实之处，本人愿意承担相关法律责任。\par
  \vspace{15mm}
  \hfill 学位论文作者签名：\hspace{4cm}\par
  \hfill 日期：\hspace{2cm}年\hspace{1cm}月\hspace{1cm}日\par

  \cleardoublepage
  \begin{center}
    {\zihao{-2}\heiti\bfseries 学位论文审查认定书}
  \end{center}
  \vspace{12pt}
  \par\zihao{-4}\kaishu
  研究生\ \underline{\hspace{3cm}}\ 在规定的学习年限内，按照培养方案及个人培养计划，完成了课程学习，成绩合格，修满规定学分；在我的指导下完成本学位论文，论文中的观点、数据、表述和结构为我所认同，论文撰写格式符合学校的相关规定，同意将本论文作为申请学位论文。\par
  \vspace{15mm}
  \hfill 导师签名：\hspace{4cm}\par
  \hfill 日期：\hspace{2cm}年\hspace{1cm}月\hspace{1cm}日\par
}

% ==================== 前置、主体、后置 ====================
\renewcommand{\frontmatter}{%
  \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{Roman}
  \pagestyle{sdust@headings}
}

\renewcommand{\mainmatter}{%
  \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}
  \pagestyle{sdust@headings}
  % 仅确保正文字号和缩进正确
  % \zihao{-4} 会根据 linespread=1.2 自动计算正确的 \baselineskip
  \zihao{-4}%
  \setlength{\parindent}{2em}%
  % 注意：学位论文通常要求两端对齐（默认），如需左对齐请取消下行注释
  % \raggedright
}

\renewcommand{\backmatter}{%
  \cleardoublepage
  \@mainmatterfalse
  \pagestyle{sdust@headings}
}

% ==================== 附录 ====================
\renewcommand{\appendix}{%
  \cleardoublepage
  \setcounter{chapter}{0}%
  \renewcommand{\thechapter}{A\arabic{chapter}}%
  \ctexset{chapter/name={附录,}}%
}

% ==================== 变量注释表环境 ====================
\newenvironment{variablelist}{%
  \chapter*{变量注释表}
  \addcontentsline{toc}{chapter}{变量注释表}
  \begin{longtable}{p{2.5cm}p{6.5cm}p{2cm}}
    \hline
    \textbf{变量} & \textbf{注释} & \textbf{初现页} \\
    \hline
    \endhead
    \hline
    \multicolumn{3}{r}{（续）} \\
    \hline
    \endfoot
    \hline
    \endlastfoot
}{%
  \end{longtable}
}

% ==================== 学位论文数据集环境 ====================
% 注意：此环境为固定表格结构，请勿在 \begin{dataset} 与 \end{dataset} 之间插入任何自定义内容
\newenvironment{dataset}{%
  \cleardoublepage
  \begin{center}
    {\zihao{-2}\heiti\bfseries 学位论文数据集}
  \end{center}
  \vspace{5mm}
  \begin{table}[H]
    \centering
    \small
    \begin{tabular}{|p{3cm}|p{3cm}|p{3cm}|p{3cm}|}
      \hline
      关键词* & 密级* & 中图分类号* & UDC \\
      \hline
      \sdust@infoget{zh/keywords} & \sdust@infoget{secretlevel} & \sdust@infoget{ccode} & \sdust@infoget{udc} \\
      \hline
      \multicolumn{4}{|l|}{论文资助：\sdust@infoget{fund}} \\
      \hline
      \multicolumn{4}{|l|}{学位授予单位名称*：山东科技大学} \\
      \multicolumn{4}{|l|}{学位授予单位代码*：10424} \\
      \hline
      学位类别* & \multicolumn{3}{l|}{\sdust@infoget{degreetype}} \\
      学位级别* & \multicolumn{3}{l|}{\sdust@infoget{degreelevel}} \\
      \hline
      论文题名* & \multicolumn{3}{l|}{\sdust@infoget{zh/title}} \\
      并列题名* & \multicolumn{3}{l|}{\sdust@infoget{en/title}} \\
      论文语种* & \multicolumn{3}{l|}{中文} \\
      \hline
      作者姓名* & \sdust@infoget{zh/author} & 学号* & \sdust@infoget{id} \\
      \hline
      培养单位名称* & \multicolumn{3}{l|}{山东科技大学} \\
      培养单位代码* & \multicolumn{3}{l|}{10424} \\
      培养单位地址 & \multicolumn{3}{l|}{山东省青岛市黄岛区前湾港路579号} \\
      邮编 & \multicolumn{3}{l|}{266590} \\
      \hline
      学科专业* & \sdust@infoget{zh/major} & 研究方向* & \sdust@infoget{researchfield} \\
      学制* & \sdust@infoget{eduduration} & 学位授予年* & \sdust@infoget{graduyear} \\
      \hline
      论文提交日期* & \sdust@infoget{submitdate} \\
      导师姓名* & \sdust@infoget{zh/supervisor} & 职称* & \sdust@infoget{supertitle} \\
      \hline
      评阅人 & \multicolumn{3}{l|}{\sdust@infoget{reviewers}} \\
      答辩委员会主席* & \sdust@infoget{defensechair} & 答辩委员会成员 & \sdust@infoget{defensecommittee} \\
      \hline
      \multicolumn{4}{|l|}{电子版论文提交格式：文本( ) 图像( ) 视频( ) 音频( ) 多媒体( ) 其他( )} \\
      \multicolumn{4}{|l|}{推荐格式：application/msword；application/pdf} \\
      \hline
      电子版论文出版（发布）者 & 电子版论文出版（发布）地 & \multicolumn{2}{l|}{权限声明} \\
      \hline
      \multicolumn{4}{|l|}{论文总页数*：\sdust@infoget{totalpages}} \\
      \hline
    \end{tabular}
  \end{table}
  \vfill
}{}

% ==================== 摘要环境 ====================
\def\@zh{zh}
\newenvironment{abstract}[1][zh]{%
  \cleardoublepage
  \def\abstract@lang{#1}%
  \ifx\abstract@lang\@zh
    \chapter*{摘要}%
    \addcontentsline{toc}{chapter}{摘要}%
  \else
    \chapter*{Abstract}%
    \addcontentsline{toc}{chapter}{Abstract}%
  \fi
  \thispagestyle{empty}%
}{%
  \par\vspace{10bp}%
  \noindent\textbf{%
    \ifx\abstract@lang\@zh 关键词：\else Keywords:\fi
  }%
  \ifx\abstract@lang\@zh
    \sdust@infoget{zh/keywords}%
  \else
    \sdust@infoget{en/keywords}%
  \fi
}

% ==================== 常用命令定义 ====================
\providecommand{\pkg}[1]{\textsf{#1}}
\providecommand{\cs}[1]{\texttt{\textbackslash #1}}
\providecommand{\BibLaTeX}{\textsc{Bib}\LaTeX}
\providecommand{\XeTeX}{\hologo{XeTeX}}
\providecommand{\email}[1]{\href{mailto:#1}{\nolinkurl{#1}}}
\providecommand{\dd}{\mathop{}\!\mathrm{d}}
\providecommand{\ee}{\mathrm{e}}
\providecommand{\ii}{\mathrm{i}}
\providecommand{\jj}{\mathrm{j}}
\providecommand{\uppi}{\pi}

\endinput
