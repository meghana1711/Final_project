BOILERPLATE_PATTERNS = [
    r'^\s*(navigation|menu|home|back to top|skip to content)\s*$',
    r'^\s*version\s+[\d.]+\s*$',
    r'^\s*v\d+\.\d+\s*$',
    r'^\s*release\s+\d+\s*$',
    r'^\s*(last modified|last updated|modified:|updated:|date:)\b.*$',
    r'^\s*\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s*$',
    r'^\s*\d{4}-\d{2}-\d{2}\s*$',
    r'^\s*page\s+\d+\s*$',
    r'^\s*chapter\s+\d+\s*$',
    r'^\s*\d+\s*$',
    r'^\s*(contents|table of contents)\s*$',
    r'^\s*(references|bibliography)\s*$',
    r'^\s*appendix\s+[a-z]\s*$',
    r'^\s*(figure|table)\s+\d+\s*$',
    r'^\s*©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?\s*$',
    r'^\s*(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$',
    r'copyright\s+©',
    r'©\s*\d{4}',
    r'\ball rights reserved\b',
    r'^\s*\[?\d+\]?\s*$',
    r'^\s*(see also:?|retrieved from|available at:|more information)\b.*$',
    r'^\s*[\*_\-=#{3,}]+\s*$',
    r'^\s*(about|overview|using|installing|get(ting)? help|get(ting)? started|documentation|mailing lists?|support( and training)?|training|troubleshooting|faq|faqs|publications?|downloads?|installation guide|release notes|changelog|related software)\s*$',
    r'^\s*slurm workload manager\s*$',
    r'^\s*schedmd\s*$',
]

PRESERVE_PATTERNS = [
    r'^\s*(?:int|void|char|const|static|extern|uint\d+_t|bool|float|double|struct|enum|typedef|union|long|short|unsigned|signed)\b',
    r'^\s*[a-z_][a-z0-9_]*\s*\([^)]*\)\s*;?$',
    r'\b[a-z_][a-z0-9_]*\s*\([^)]*\)\s*;?$',
    r'^\s*[A-Z_][A-Z0-9_]+\b',
    r'^\s*#\s*define\b',
    r'^\s*#\s*include\b',
    r'^\s*(arguments?|returns?|description|parameters?|example|note|warning|syntax|usage|input|output|overview):\s*$',
    r'^\s*(api\s+(?:functions|methods|calls)|function\s+(?:reference|list)|method\s+(?:reference|list))\s*$',
    r'\b(?:SLURM_SUCCESS|SLURM_ERROR|SUCCESS|ERROR|FAILURE|OK)\b',
    r'\([^)]*(?:input|output|in|out|inout)[^)]*\)',
    r'^\s*//',
    r'^\s*/\*',
    r'\*/\s*$',
]       

ABBREVIATIONS = {
    'e.g.', 'i.e.', 'etc.', 'vs.', 'cf.', 'inc.', 'ltd.', 'corp.',
    'fig.', 'figs.', 'eq.', 'eqs.', 'sec.', 'secs.', 'ch.', 'chs.',
    'vol.', 'vols.', 'no.', 'nos.', 'p.', 'pp.', 'ref.', 'refs.',
    'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'rev.', 'gen.', 'col.',
    'maj.', 'capt.', 'lt.', 'sgt.', 'pvt.', 'jr.', 'sr.',
    'min.', 'max.', 'avg.', 'std.', 'var.', 'cpu.', 'gpu.',
    'api.', 'url.', 'uri.', 'sql.', 'xml.', 'json.', 'csv.',
    'kb.', 'mb.', 'gb.', 'tb.', 'hz.', 'mhz.', 'ghz.'
}

NON_BOUNDARY = [
    r'\b[A-Z][a-z]*\.\s+[a-z]',  # "Inc. and"
    r'\b\d+\.\s*\d',             # "3.14"
    r'\b[A-Z]\.\s*[A-Z]\.',      # "U.S.A."
    r'Fig\.\s*\d+',
    r'Table\s*\d+\.',
    r'Section\s*\d+\.',
    r'Chapter\s*\d+\.'
]

FRAGMENT_PATTERNS = [
    r'^\w{1,3}$',
    r'^[A-Z]+$',
    r'^Figure\s*\d*\.?$',
    r'^Table\s*\d*\.?$',
    r'^Example\s*\d*\.?$',
    r'^Section\s*\d*\.?$',
    r'^Chapter\s*\d*\.?$',
    r'^\d+\.?\s*$',
    r'^[a-z]$',
    r'^\([^)]*\)$',
    r'^[\[\]{}().,;:!?-]+$',
    r'^none$',
    r'^yes$',
    r'^no$',
]

SPLIT_PATTERNS = [
    r'([.!?]\s+)(?=[A-Z])',
    r'(;\s+)(?=[A-Z])',
    r'(:\s+)(?=[A-Z][a-z])',
    r'(\n\n+)',
    r'(\.\s+)(?=\d+\.)',
]

STRUCTURAL_HEADERS = {
    'api functions', 'api', 'functions', 'methods',
    'description', 'arguments', 'returns', 'parameters',
    'examples', 'example', 'syntax', 'usage',
    'notes', 'note', 'warnings', 'warning',
    'input', 'output', 'configuration', 'options', 'specifications'
}