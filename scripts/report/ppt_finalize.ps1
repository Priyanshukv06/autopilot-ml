# Adds the slide transitions and entrance animations to the presentation, saves it and exports a PDF copy.
# Shapes named "anim:NN:effect" (effect: fade or wipe) are animated in the order NN, each starting
# automatically after the previous one, so every slide needs a single click. Transitions: Fade, 0.6 s.
param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)
$ErrorActionPreference = "Stop"
$effects = @{ "fade" = 10; "wipe" = 22 }   # msoAnimEffectFade, msoAnimEffectWipe
$afterPrevious = 3                          # msoAnimTriggerAfterPrevious
$fromLeft = 4                               # msoAnimDirectionLeft
$fadeSmoothly = 3849                        # ppEffectFadeSmoothly

# PowerPoint has a single instance: if it is already open, use it but never quit it.
$wasRunning = @(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$app = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pres = $app.Presentations.Open($InPath, 0, 0, 0)   # read-write, no window
    $count = 0
    foreach ($slide in $pres.Slides) {
        $slide.SlideShowTransition.EntryEffect = $fadeSmoothly
        $slide.SlideShowTransition.Duration = 0.6
        $items = @()
        foreach ($shape in $slide.Shapes) {
            if ($shape.Name -like "anim:*") { $items += $shape }
        }
        $items = $items | Sort-Object { [int]($_.Name.Split(":")[1]) }
        foreach ($shape in $items) {
            $kind = $shape.Name.Split(":")[2]
            $effect = $slide.TimeLine.MainSequence.AddEffect($shape, $effects[$kind], 0, $afterPrevious)
            $effect.Timing.Duration = 0.5
            if ($kind -eq "wipe") { $effect.EffectParameters.Direction = $fromLeft; $effect.Timing.Duration = 0.7 }
            $count += 1
        }
    }
    $pres.Save()
    $pres.SaveCopyAs($PdfPath, 32)   # ppSaveAsPDF
    $slides = $pres.Slides.Count
    $pres.Close()
    $pres = $null
    Write-Output "slides=$slides animations=$count pdf=$PdfPath"
}
finally {
    if ($null -ne $pres) { $pres.Close() }
    if (-not $wasRunning -and $app.Presentations.Count -eq 0) { $app.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
