# Open a .docx in Microsoft Word, optionally update every field (page references,
# caption numbers, page numbers) and save it, and optionally export it to PDF.
# Used by scripts/report/build_report.py; needs Microsoft Word on Windows.
param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [string]$PdfPath = "",
    [switch]$UpdateFields,
    [switch]$Save
)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $readOnly = -not $Save
    $doc = $word.Documents.Open($InPath, $false, $readOnly)
    if ($UpdateFields) {
        # Two passes: page numbers can move once the lists at the front are filled in.
        for ($pass = 0; $pass -lt 2; $pass++) {
            $doc.Fields.Update() | Out-Null
            foreach ($section in $doc.Sections) {
                foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
                foreach ($header in $section.Headers) { $header.Range.Fields.Update() | Out-Null }
            }
            $doc.Repaginate()
        }
    }
    if ($Save) { $doc.Save() }
    if ($PdfPath -ne "") {
        # PDF for printing, with bookmarks made from the headings.
        $doc.ExportAsFixedFormat($PdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    }
    $pages = $doc.ComputeStatistics(2)
    Write-Output "pages=$pages"
    $doc.Close($false)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
