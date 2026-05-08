#!/usr/bin/env python3
"""
annotate_pdf_final.py  —  UPSC answer-sheet annotation engine
Usage:  python3 annotate_pdf_final.py input.pdf remarks.json output.pdf [dpi]
        dpi default = 250

Teacher-authentic rendering:
  • Wavy underlines  — follow actual ink-bottom profile column-by-column
  • Organic circles  — wobbly ellipse with Fourier harmonics + slight tilt
  • Natural tick ✓   — two curved Bezier strokes like pen on paper
  • Organic arrows   — quadratic Bezier through free margin bands
  • Proximity layout — remarks placed at annotation height first
"""

import sys, json, math, textwrap, os, tempfile, random
import numpy as np, cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import fitz  # PyMuPDF — raster fallback when poppler (pdfinfo) is unavailable
import img2pdf


def _load_pdf_pages_pil(pdf_in: str, dpi: int) -> list:
    """RGB PIL pages at ``dpi``. Prefers poppler via pdf2image; falls back to PyMuPDF."""
    try:
        return convert_from_path(pdf_in, dpi=dpi)
    except Exception:
        doc = fitz.open(pdf_in)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pages: list[Image.Image] = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            if pix.n == 4:
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                img = img.convert("RGB")
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()
        return pages

RED      = (190, 20,  20)
RED_SOFT = (210, 55,  55)
GREEN    = ( 15, 130, 45)
SEV      = {"critical": RED, "suggestion": RED_SOFT, "positive": GREEN}
def col(r): return SEV.get(r.get("severity","suggestion"), RED_SOFT)
def bgr(c): return (c[2],c[1],c[0])

# ── font — Homemade Apple only (Google Fonts) ────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(_DIR, "fonts")


def _font_path() -> str:
    """Resolve Homemade Apple; no other typefaces are used for remark raster text."""
    candidates = [
        os.path.join(_FONTS_DIR, "HomemadeApple-Regular.ttf"),
        os.path.join(_DIR, "HomemadeApple-Regular.ttf"),
        os.path.join(_DIR, "HomemadeApple.ttf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Homemade Apple not found. Install "
        "https://fonts.google.com/specimen/Homemade+Apple "
        f"as HomemadeApple-Regular.ttf under {_FONTS_DIR!r} or {_DIR!r}."
    )


FONT_PATH = _font_path()
IS_HW = True  # only Homemade Apple — no synthetic slant overlay
print(f"[font] {os.path.basename(FONT_PATH)}")

def pil_font(sz): return ImageFont.truetype(FONT_PATH, max(8,sz))
def measure(text,sz):
    d=ImageDraw.Draw(Image.new("L",(1,1)))
    bb=d.textbbox((0,0),text,font=pil_font(sz)); return bb[2]-bb[0],bb[3]-bb[1]
def hw_img(text,sz,rgba):
    fnt=pil_font(sz); tw,th=measure(text,sz); pad=max(4,sz//4)
    img=Image.new("RGBA",(tw+2*pad,th+2*pad),(0,0,0,0))
    ImageDraw.Draw(img).text((pad,pad),text,font=fnt,fill=rgba)
    if not IS_HW:
        arr=np.array(img); M=np.float32([[1,0.22,0],[0,1,0]])
        W2=int(img.width+img.height*0.22)
        arr=cv2.warpAffine(arr,M,(W2,img.height),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        img=Image.fromarray(arr)
    return img
def paste_hw(layer,text,x,y,sz,color):
    img=hw_img(text,sz,color+(255,))
    layer.paste(img,(int(x),int(y)),img); return img.size

# ═══════════════════════════════════════════════════════════════════════════════
#  FREE-FORM DRAWING PRIMITIVES  (all teacher-authentic)
# ═══════════════════════════════════════════════════════════════════════════════

def _seed(x,y): random.seed(int(x)*31337+int(y))

# ── 1. WAVY UNDERLINE following text baseline ─────────────────────────────────
def wavy_underline(arr, gray, line_y, lh, x1, x2, color, thickness, dpi, pad_px=4):
    """
    Draw an underline whose y-profile follows the actual ink-bottom of the text.
    Algorithm:
      1. Extract per-column ink bottom in the line band
      2. Interpolate gaps (word spaces have no ink)
      3. Gaussian-smooth to get flowing wave (σ ∝ dpi)
      4. Draw polyline slightly below the smoothed profile
    """
    x1c = max(0,x1); x2c = min(gray.shape[1],x2)
    if x2c <= x1c: return

    y_top = max(0, line_y-lh)
    y_bot = min(gray.shape[0], line_y+lh+int(lh*0.15))  # tight: avoid next line
    band  = gray[y_top:y_bot, x1c:x2c]
    blur  = cv2.GaussianBlur(band,(3,3),0)
    bw    = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV,31,10)

    # Per-column: find bottommost ink row
    bottoms = []
    for c in range(bw.shape[1]):
        nz = np.nonzero(bw[:,c])[0]
        bottoms.append(float(y_top+nz[-1]) if len(nz)>0 else None)

    # Forward + backward fill gaps
    filled = bottoms[:]
    last = None
    for i,v in enumerate(filled):
        if v is not None: last=v
        elif last is not None: filled[i]=last
    last=None
    for i in range(len(filled)-1,-1,-1):
        if filled[i] is not None: last=filled[i]
        elif last is not None: filled[i]=last
    if all(v is None for v in filled):
        # fallback: flat line
        y_flat = line_y+lh+pad_px
        cv2.line(arr,(x1c,y_flat),(x2c,y_flat),bgr(color),thickness,cv2.LINE_AA)
        return
    arr_f = np.array([v if v is not None else line_y+lh for v in filled], dtype=float)

    # Sliding-window 85th-percentile with large window = ignore descenders robustly
    win = max(40, int(100*dpi/150))
    robust = np.zeros_like(arr_f)
    for i in range(len(arr_f)):
        w0=max(0,i-win//2); w1=min(len(arr_f),i+win//2)
        window_vals=[arr_f[j] for j in range(w0,w1) if filled[j] is not None]
        robust[i] = np.percentile(window_vals,82) if window_vals else arr_f[i]

    # Heavy Gaussian smooth → gentle flowing wave, no letter-level noise
    sigma = max(25.0, 55.0*dpi/150)
    smoothed = gaussian_filter1d(robust, sigma=sigma) + pad_px

    # Enforce: underline must not go above actual ink bottom
    for i in range(len(smoothed)):
        if filled[i] is not None:
            smoothed[i] = max(smoothed[i], filled[i]+1)

    # Subsample every 4px for a smooth polyline
    xs = list(range(x1c, x2c, 4))
    ys = [int(smoothed[min(j, len(smoothed)-1)]) for j in range(0, x2c-x1c, 4)]
    pts = np.array(list(zip(xs,ys)), np.int32).reshape(-1,1,2)
    cv2.polylines(arr,[pts],False,bgr(color),thickness,cv2.LINE_AA)


# ── 2. ORGANIC ELLIPSE (wobbly, like hand-drawn circle) ───────────────────────
def organic_ellipse(arr, cx, cy, rx, ry, color, thickness, dpi):
    """
    Draw a slightly wobbly ellipse.
    Algorithm:
      • Parametric ellipse with Fourier perturbation on the radius
      • Two harmonics give the 'not quite round' teacher feel
      • Slight tilt makes it more natural
      • Starts slightly past 0° and runs ~370° (slight overrun = pen continuation)
    """
    _seed(cx,cy)
    tilt   = math.radians(random.uniform(-8, 8))
    wobble = 0.045 + random.uniform(0, 0.02)
    h1_amp = wobble;   h1_ph = random.uniform(0, 2*math.pi)
    h2_amp = wobble*0.55; h2_ph = random.uniform(0, 2*math.pi)
    h3_amp = wobble*0.3;  h3_ph = random.uniform(0, 2*math.pi)

    start_ang = random.uniform(-0.18, 0.18)   # start slightly off 0
    total_ang  = 2*math.pi + random.uniform(0.05, 0.18)  # slight overrun
    N = max(80, int(120*dpi/150))
    pts = []
    for i in range(N+1):
        t = start_ang + total_ang*i/N
        rw = (1 + h1_amp*math.sin(5*t+h1_ph)
                + h2_amp*math.sin(11*t+h2_ph)
                + h3_amp*math.sin(17*t+h3_ph))
        xp = rx*rw*math.cos(t)
        yp = ry*rw*math.sin(t)
        xr = xp*math.cos(tilt)-yp*math.sin(tilt)
        yr = xp*math.sin(tilt)+yp*math.cos(tilt)
        pts.append((int(cx+xr), int(cy+yr)))
    p = np.array(pts,np.int32).reshape(-1,1,2)
    cv2.polylines(arr,[p],False,bgr(color),thickness,cv2.LINE_AA)


# ── 3. NATURAL TICK MARK (two curved Bezier strokes) ──────────────────────────
def natural_tick(arr, cx, cy, size, color, thickness):
    """
    Draw a ✓ tick as two curved Bezier strokes.
    Stroke 1: short descending-left  (the small foot of the tick)
    Stroke 2: long ascending-right   (the main arm, with forward lean)
    Both strokes have slight curvature for organic feel.
    """
    _seed(cx,cy)

    def bezier_pts(p0, cp, p1, n=30):
        pts=[]
        for i in range(n+1):
            t=i/n
            x=int((1-t)**2*p0[0]+2*(1-t)*t*cp[0]+t**2*p1[0])
            y=int((1-t)**2*p0[1]+2*(1-t)*t*cp[1]+t**2*p1[1])
            pts.append((x,y))
        return pts

    s = size
    # Small foot: from upper-left down to junction
    p0_s1 = (cx-s//2, cy-s//8)
    p1_s1 = (cx-s//8, cy+s//2)
    cp_s1 = (cx-s//2+int(s*0.15)+random.randint(-2,2),
             cy+s//8+random.randint(-2,2))
    # Long arm: from junction up-right
    p0_s2 = p1_s1
    p1_s2 = (cx+s//2+random.randint(-3,3), cy-s//2+random.randint(-3,3))
    cp_s2 = (cx+int(s*0.1)+random.randint(-2,2),
             cy+int(s*0.2)+random.randint(-2,2))

    pts1 = np.array(bezier_pts(p0_s1,cp_s1,p1_s1,20),np.int32).reshape(-1,1,2)
    pts2 = np.array(bezier_pts(p0_s2,cp_s2,p1_s2,40),np.int32).reshape(-1,1,2)
    cv2.polylines(arr,[pts1],False,bgr(color),thickness,cv2.LINE_AA)
    cv2.polylines(arr,[pts2],False,bgr(color),thickness,cv2.LINE_AA)


# ── 4. ORGANIC ARROW through free margin bands ────────────────────────────────
def organic_arrow(arr, sx, sy, dx, dy, color, th, rm_free_bands, H):
    """Smooth quadratic Bezier arrow, bowing through free space."""
    dist = math.hypot(dx-sx,dy-sy)
    if dist<5: return
    going_right = dx > sx+50
    going_down  = dy > sy+80

    if going_right:
        bow = min(dist*0.32, H*0.055)
        cpx = (sx+dx)//2
        cpy = min(sy,dy)-int(bow)
        if dy>sy: cpy = sy-int(bow*0.55)
    elif going_down:
        bow = min(dist*0.28, H*0.05)
        cpx = max(sx,dx)+int(bow)
        cpy = (sy+dy)//2
    else:
        angle=math.atan2(dy-sy,dx-sx); perp=angle+math.pi/2
        bow=dist*0.28
        cpx=int((sx+dx)//2+bow*math.cos(perp)); cpy=int((sy+dy)//2+bow*math.sin(perp))

    N=100; pts=[]
    for i in range(N+1):
        t=i/N
        x=int((1-t)**2*sx+2*(1-t)*t*cpx+t**2*dx)
        y=int((1-t)**2*sy+2*(1-t)*t*cpy+t**2*dy)
        pts.append((x,y))
    p=np.array(pts,np.int32).reshape(-1,1,2)
    cv2.polylines(arr,[p],False,bgr(color),th,cv2.LINE_AA)
    # Arrowhead
    ang=math.atan2(pts[-1][1]-pts[-4][1],pts[-1][0]-pts[-4][0])
    alen=max(14,th*6)
    for da in [0.45,-0.45]:
        ax=int(dx-alen*math.cos(ang+da)); ay=int(dy-alen*math.sin(ang+da))
        cv2.line(arr,(dx,dy),(ax,ay),bgr(color),th,cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECISION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_occupancy(gray,body_l,body_r,rm_start):
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    H=gray.shape[0]
    body_occ=np.array([bw[y,body_l:body_r].sum()>200 for y in range(H)])
    rm_occ  =np.array([bw[y,rm_start:].sum()>50   for y in range(H)])
    return body_occ,rm_occ

def free_bands(occ,min_h=30):
    bands=[]; in_f=False; start=0
    for y,occ_y in enumerate(occ):
        if not occ_y and not in_f: start=y; in_f=True
        elif occ_y and in_f:
            if y-start>=min_h: bands.append((start,y))
            in_f=False
    if in_f and len(occ)-start>=min_h: bands.append((start,len(occ)))
    return bands

def find_ink_bottom(gray,line_y,lh,x1,x2,pad=4):
    y_top=max(0,line_y-lh); y_bot=min(gray.shape[0],line_y+lh+int(lh*0.4))
    x1c=max(0,x1); x2c=min(gray.shape[1],x2)
    band=gray[y_top:y_bot,x1c:x2c]
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(band,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    rs=bw.sum(axis=1); thr=max(rs)*0.06 if max(rs)>0 else 1
    last=0
    for i in range(len(rs)-1,-1,-1):
        if rs[i]>thr: last=i; break
    return y_top+last+pad

def find_ink_top(gray,line_y,lh,x1,x2):
    y_top=max(0,line_y-lh); y_bot=min(gray.shape[0],line_y+lh)
    x1c=max(0,x1); x2c=min(gray.shape[1],x2)
    band=gray[y_top:y_bot,x1c:x2c]
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(band,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    rs=bw.sum(axis=1); thr=max(rs)*0.06 if max(rs)>0 else 1
    for i in range(len(rs)):
        if rs[i]>thr: return y_top+i
    return line_y

def snap_circle_words(gray,line_y,lh,body_l,body_r,x_start,x_end,dpi):
    y1=max(0,line_y-lh); y2=min(gray.shape[0],line_y+lh)
    band=gray[y1:y2,body_l:body_r]
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(band,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    mk=int(20*dpi/150)
    dil=cv2.dilate(bw,cv2.getStructuringElement(cv2.MORPH_RECT,(mk,1)))
    c=dil.sum(axis=0).astype(float)
    if c.max()==0: return x_start,x_end
    thresh=c.max()*0.08; in_w=False; ws=0; words=[]
    for x2,v in enumerate(c):
        if v>thresh and not in_w: ws=x2; in_w=True
        elif v<=thresh and in_w:
            if x2-ws>int(15*dpi/150):
                raw=bw[:,ws:x2].sum(axis=0).astype(float)
                ol=next((i for i,v2 in enumerate(raw) if v2>0),0)
                or_=next((i for i,v2 in enumerate(reversed(raw)) if v2>0),0)
                words.append((ws+ol+body_l,x2-or_+body_l))
            in_w=False
    if in_w: words.append((ws+body_l,len(c)+body_l))
    if not words: return x_start,x_end
    ov=[(a,b) for a,b in words if a<x_end and b>x_start]
    if not ov: return x_start,x_end
    return min(a for a,b in ov),max(b for a,b in ov)

def snap_exponent_gap(gray,line_y,lh,body_l,body_r,target_x,dpi):
    y1=max(0,line_y-lh); y2=min(gray.shape[0],line_y+lh)
    band=gray[y1:y2,body_l:body_r]
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(band,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    mk=int(20*dpi/150)
    dil=cv2.dilate(bw,cv2.getStructuringElement(cv2.MORPH_RECT,(mk,1)))
    c=dil.sum(axis=0).astype(float)
    if c.max()==0: return target_x
    thresh=c.max()*0.08; in_w=False; ws=0; words=[]
    for x2,v in enumerate(c):
        if v>thresh and not in_w: ws=x2; in_w=True
        elif v<=thresh and in_w:
            if x2-ws>int(15*dpi/150): words.append((ws+body_l,x2+body_l))
            in_w=False
    if in_w: words.append((ws+body_l,len(c)+body_l))
    if len(words)<2: return target_x
    gaps=[(words[i][1]+words[i+1][0])//2 for i in range(len(words)-1)]
    return min(gaps,key=lambda g:abs(g-target_x))

# ── line / gap detection ──────────────────────────────────────────────────────
def detect_lines(pil_img,dpi):
    img=np.array(pil_img.convert('L'))
    bw=cv2.adaptiveThreshold(cv2.GaussianBlur(img,(3,3),0),255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,10)
    sc=dpi/150.0; n,_,stats,centroids=cv2.connectedComponentsWithStats(bw)
    char_ys=sorted(centroids[i][1] for i in range(1,n)
        if(int(12*sc)<stats[i][3]<int(90*sc) and
           int(5*sc)<stats[i][2]<int(150*sc) and
           int(40*sc**2)<stats[i][4]<int(8000*sc**2)))
    if not char_ys: return []
    cg=int(22*sc); mt=int(35*sc); raw,b=[],[char_ys[0]]
    for y in char_ys[1:]:
        if y-b[-1]<cg: b.append(y)
        else:
            if len(b)>=3: raw.append(int(np.median(b)))
            b=[y]
    if len(b)>=3: raw.append(int(np.median(b)))
    merged=[raw[0]]
    for y in raw[1:]:
        if y-merged[-1]>mt: merged.append(y)
    return merged

def detect_gaps(line_ys,H,lh,min_gap):
    gaps=[]
    for i in range(len(line_ys)-1):
        gt=line_ys[i]+lh+4; gb=line_ys[i+1]-lh-4
        if gb-gt>=min_gap:
            gaps.append(dict(after_line=i+1,y_top=gt,y_bot=gb,
                             height=gb-gt,y_center=(gt+gb)//2))
    gt=line_ys[-1]+lh+4
    if H-4-gt>=min_gap: gaps.append(dict(after_line=len(line_ys),y_top=gt,
                                         y_bot=H-4,height=H-4-gt,y_center=(gt+H-4)//2))
    return gaps

def find_gap_near(line_num,gaps,prefer='below',max_dist=5):
    below=[g for g in gaps if line_num<=g['after_line']<=line_num+max_dist]
    above=[g for g in gaps if line_num-max_dist<=g['after_line']<line_num]
    if prefer=='below' and below: return min(below,key=lambda g:g['after_line'])
    if above: return max(above,key=lambda g:g['after_line'])
    if below: return min(below,key=lambda g:g['after_line'])
    return None

# ── proximity layout ──────────────────────────────────────────────────────────
class ProximityLayout:
    def __init__(self,W,H,LMX,RMX):
        self.W=W; self.H=H; self.LMX=LMX; self.RMX=RMX
        self._slots={'left_margin':[],'right_margin':[]}
    def zone_x(self,z): return 4 if z=="left_margin" else self.RMX+6
    def zone_w(self,z): return self.LMX-8 if z=="left_margin" else self.W-self.RMX-10
    def _ok(self,zone,y0,y1,m=6):
        for pt,pb in self._slots[zone]:
            if not(y1+m<=pt or y0-m>=pb): return False
        return True
    def find_y(self,zone,pref,h,m=6):
        if self._ok(zone,pref,pref+h,m):
            self._slots[zone].append((pref,pref+h)); self._slots[zone].sort(); return pref
        step=max(4,h//6)
        for off in range(step,self.H,step):
            for d in[-1,1]:
                y=pref+d*off
                if y<2 or y+h>self.H-2: continue
                if self._ok(zone,y,y+h,m):
                    self._slots[zone].append((y,y+h)); self._slots[zone].sort(); return y
        y=self._slots[zone][-1][1]+m if self._slots[zone] else pref
        self._slots[zone].append((y,y+h)); return y
    def render(self,layer,lines_str,zone,preferred_y,sz,color):
        zw=self.zone_w(zone); zx=self.zone_x(zone)
        lh=measure("A",sz)[1]+max(2,sz//8)
        cw=max(1,int(zw/max(1,measure("m",sz)[0])))
        rows=[]
        for ln in lines_str: rows+=textwrap.wrap(ln,cw) or [ln]
        total=len(rows)*lh
        y0=self.find_y(zone,preferred_y-total//2,total)
        for i,row in enumerate(rows): paste_hw(layer,row,zx,y0+i*lh,sz,color)
        return zx,y0,total

def place_in_gap(layer,text_lines,gap,body_l,sz,color,body_r=None):
    avail_w=(body_r-body_l-30) if body_r else 1500
    cw=max(1,int(avail_w/max(1,measure("m",sz)[0])))
    rows=[]
    for ln in text_lines: rows+=textwrap.wrap(ln,cw) or [ln]
    lh=measure("A",sz)[1]+max(3,sz//7)
    tot=len(rows)*lh
    y0=gap['y_top']+(gap['height']-tot)//2 if tot<gap['height']-8 else gap['y_top']+4
    x0=body_l+30
    for i,row in enumerate(rows): paste_hw(layer,row,x0,y0+i*lh,sz,color)
    return x0,y0

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANNOTATOR
# ═══════════════════════════════════════════════════════════════════════════════
def annotate_page(pil_img,remarks,line_ys,dpi):
    W,H=pil_img.size; gray=np.array(pil_img.convert('L')); sc=dpi/150.0
    LH=max(12,int(H//55)); LMX=int(W*0.135); RMX=int(W*0.875)
    LN_TH=max(2,int(2*sc)); TICK_SZ=max(10,int(H//40)); TICK_TH=max(2,int(3*sc))
    TIP_W=max(10,int(18*sc)); ARR_TH=max(2,int(2.5*sc))
    S_REM=max(26,int(H//65)); S_MAR=max(18,int(H//100))
    S_INS=max(20,int(H//85)); S_SCR=max(36,int(H//44))

    _,rm_occ=build_occupancy(gray,LMX,RMX,RMX)
    rm_free=free_bands(rm_occ,min_h=int(40*sc))
    gaps=detect_gaps(line_ys,H,LH,max(40,int(55*sc)))
    layout=ProximityLayout(W,H,LMX,RMX)

    def ly(n): return line_ys[max(0,min(n-1,len(line_ys)-1))]

    arr=cv2.cvtColor(np.array(pil_img.convert("RGB")),cv2.COLOR_RGB2BGR)
    text_l=Image.new("RGBA",(W,H),(0,0,0,0))

    PRIO={"score_circle":0,"tick":1,"circle_remark":2,"underline_remark":3,
          "exponent_insert":4,"arrow_remark":5,"curly_brace":6,"text":7}
    def sk(r):
        for k in ("line","line_start","cy_line"):
            if k in r: return r[k]
        return 99

    for r in sorted(remarks,key=lambda r:(PRIO.get(r["type"],9),sk(r))):
        t=r["type"]; c=col(r)

        # ── score_circle ──────────────────────────────────────────────────────
        if t=="score_circle":
            cx=int(W*r["cx"]/100)
            cy=ly(r["cy_line"]) if "cy_line" in r else int(H*r.get("cy",5)/100)
            txt=str(r["text"]); tw,th2=measure(txt,S_SCR)
            rad=max(tw,th2)//2+max(16,int(20*sc))
            organic_ellipse(arr,cx,cy,rad,rad,c,max(2,int(3*sc)),dpi)
            paste_hw(text_l,txt,cx-tw//2-2,cy-th2//2-2,S_SCR,c)

        # ── tick ✓ — natural curved strokes ──────────────────────────────────
        elif t=="tick":
            cy=ly(r["line"])
            cx=int(W*0.07) if r.get("zone","left_margin")=="left_margin" else int(W*0.93)
            natural_tick(arr,cx,cy,TICK_SZ,c,TICK_TH)

        # ── circle_remark — organic oval centred on actual ink ────────────────
        elif t=="circle_remark":
            cy=ly(r["line"])
            xs=int(W*r["x_start"]/100); xe=int(W*r["x_end"]/100)
            xs,xe=snap_circle_words(gray,cy,LH,LMX,RMX,xs,xe,dpi)
            cx=(xs+xe)//2
            itop=find_ink_top(gray,cy,LH,xs,xe)
            ibot=find_ink_bottom(gray,cy,LH,xs,xe,pad=2)
            rx=(xe-xs)//2+max(6,int(8*sc))
            ry=(ibot-itop)//2+max(4,int(5*sc))
            oval_cy=(itop+ibot)//2
            organic_ellipse(arr,cx,oval_cy,rx,ry,c,LN_TH,dpi)

            remark=r.get("remark",""); zone=r.get("remark_zone","gap")
            if remark:
                lines_r=[remark]
                if zone=="gap":
                    gap=find_gap_near(r["line"],gaps,'below')
                    if gap and gap['height']>=S_REM+10:
                        tx,ty=place_in_gap(text_l,lines_r,gap,LMX,S_REM,c,RMX)
                        organic_arrow(arr,cx,oval_cy+ry,tx,ty+S_REM//2,c,ARR_TH,rm_free,H)
                    else:
                        _,y0,th_=layout.render(text_l,lines_r,"right_margin",oval_cy,S_MAR,c)
                        organic_arrow(arr,cx+rx,oval_cy,RMX+4,y0+th_//2,c,ARR_TH,rm_free,H)
                else:
                    _,y0,th_=layout.render(text_l,lines_r,zone,oval_cy,S_MAR,c)
                    ax=cx+rx if "right" in zone else cx-rx
                    bx2=RMX+4 if "right" in zone else LMX-4
                    organic_arrow(arr,ax,oval_cy,bx2,y0+th_//2,c,ARR_TH,rm_free,H)

        # ── underline_remark — wavy, follows text baseline ────────────────────
        elif t=="underline_remark":
            x1=int(W*r["x_start"]/100); x2=int(W*r["x_end"]/100)
            # Draw wavy underline following ink bottom profile
            wavy_underline(arr,gray,ly(r["line"]),LH,x1,x2,c,LN_TH,dpi)
            anchor_x=x2
            anchor_y=find_ink_bottom(gray,ly(r["line"]),LH,x1,x2)

            if "line_end" in r:
                x1b=int(W*r.get("x_start2",r["x_start"])/100)
                x2b=int(W*r.get("x_end2",  r["x_end"])/100)
                wavy_underline(arr,gray,ly(r["line_end"]),LH,x1b,x2b,c,LN_TH,dpi)
                anchor_x=x2b
                anchor_y=find_ink_bottom(gray,ly(r["line_end"]),LH,x1b,x2b)

            remark=r.get("remark",""); zone=r.get("remark_zone","gap")
            if remark:
                lines_r=[remark]; ln_num=r.get("line_end",r["line"])
                if zone=="gap":
                    gap=find_gap_near(ln_num,gaps,'below')
                    if gap and gap['height']>=S_REM+10:
                        tx,ty=place_in_gap(text_l,lines_r,gap,LMX,S_REM,c,RMX)
                        organic_arrow(arr,anchor_x,anchor_y,tx,ty+S_REM//2,c,ARR_TH,rm_free,H)
                    else:
                        _,y0,th_=layout.render(text_l,lines_r,"right_margin",anchor_y,S_MAR,c)
                        organic_arrow(arr,anchor_x,anchor_y,RMX+4,y0+th_//2,c,ARR_TH,rm_free,H)
                else:
                    _,y0,th_=layout.render(text_l,lines_r,zone,anchor_y,S_MAR,c)
                    bx2=RMX+4 if "right" in zone else LMX-4
                    organic_arrow(arr,anchor_x,anchor_y,bx2,y0+th_//2,c,ARR_TH,rm_free,H)

        # ── exponent_insert — snapped to word gap, placed above ink top ───────
        elif t=="exponent_insert":
            raw_x=int(W*r["x_pct"]/100); line_y_v=ly(r["line"])
            cx=snap_exponent_gap(gray,line_y_v,LH,LMX,RMX,raw_x,dpi)
            itop=find_ink_top(gray,line_y_v,LH,cx-20,cx+20)
            s=max(8,int(H//78)); caret_y=itop-s-6
            cv2.line(arr,(cx,caret_y),(cx-s,caret_y+s),bgr(RED_SOFT),LN_TH,cv2.LINE_AA)
            cv2.line(arr,(cx,caret_y),(cx+s,caret_y+s),bgr(RED_SOFT),LN_TH,cv2.LINE_AA)
            ins=r.get("text","")
            if ins:
                tw,th2=measure(ins,S_INS)
                paste_hw(text_l,ins,cx-tw//2,caret_y-th2-4,S_INS,RED_SOFT)

        # ── arrow_remark ──────────────────────────────────────────────────────
        elif t=="arrow_remark":
            from_x=int(W*r.get("from_x",50)/100); from_y=ly(r["from_line"])
            remark=r.get("remark",""); zone=r.get("remark_zone","right_margin")
            if remark:
                _,y0,th_=layout.render(text_l,[remark],zone,from_y,S_MAR,c)
                to_x=RMX+4 if "right" in zone else LMX-4
                organic_arrow(arr,from_x,from_y,to_x,y0+th_//2,c,ARR_TH,rm_free,H)

        # ── curly_brace ───────────────────────────────────────────────────────
        elif t=="curly_brace":
            zone=r.get("zone","left_margin")
            y_top=ly(r["line_start"])-LH; y_bot=ly(r["line_end"])+LH
            bx=LMX-max(2,int(3*sc)) if zone=="left_margin" else RMX+max(2,int(3*sc))
            side="left" if zone=="left_margin" else "right"
            Hb=y_bot-y_top; tip=bx+(TIP_W if side=="left" else -TIP_W)
            mid=y_top+Hb//2; N=80; pts=[]
            for i in range(N+1):
                t2=i/N
                bez=lambda a,b,c2,d:(1-t2)**3*a+3*(1-t2)**2*t2*b+3*(1-t2)*t2**2*c2+t2**3*d
                pts.append((int(bez(bx,tip,tip,tip)),int(bez(y_top,y_top,mid-Hb//8,mid))))
            for i in range(N+1):
                t2=i/N
                bez=lambda a,b,c2,d:(1-t2)**3*a+3*(1-t2)**2*t2*b+3*(1-t2)*t2**2*c2+t2**3*d
                pts.append((int(bez(tip,tip,tip,bx)),int(bez(mid,mid+Hb//8,y_bot,y_bot))))
            cv2.polylines(arr,[np.array(pts,np.int32).reshape(-1,1,2)],False,bgr(c),LN_TH,cv2.LINE_AA)
            if r.get("remark"):
                layout.render(text_l,[r["remark"]],zone,(y_top+y_bot)//2,S_MAR,c)

        # ── plain text ────────────────────────────────────────────────────────
        elif t=="text":
            zone=r.get("zone","right_margin")
            ay=ly(r["line"]) if "line" in r else int(H*r.get("anchor_y",50)/100)
            layout.render(text_l,r.get("text","").split("\n"),zone,ay,S_MAR,c)

    base=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).convert("RGBA")
    return Image.alpha_composite(base,text_l).convert("RGB")

# ── entry ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv)<4:
        print("Usage: python3 annotate_pdf_final.py input.pdf remarks.json output.pdf [dpi]")
        sys.exit(1)
    pdf_in,json_in,pdf_out=sys.argv[1],sys.argv[2],sys.argv[3]
    dpi=int(sys.argv[4]) if len(sys.argv)>4 else 250
    with open(json_in) as f: spec=json.load(f)
    page_map={p["page"]:p.get("remarks",[]) for p in spec.get("pages",[])}
    print(f"[+] Rendering at {dpi} DPI…")
    pages=_load_pdf_pages_pil(pdf_in, dpi=dpi)
    print(f"[+] {len(pages)} page(s)  {pages[0].size[0]}×{pages[0].size[1]}px")
    annotated=[]
    for i,img in enumerate(pages,1):
        remarks=page_map.get(i,[])
        line_ys=detect_lines(img,dpi)
        print(f"[+] Page {i}: {len(line_ys)} lines, {len(remarks)} remarks")
        annotated.append(annotate_page(img,remarks,line_ys,dpi))
    print("[+] Writing PDF…")
    with tempfile.TemporaryDirectory() as tmp:
        paths=[]
        for i,img in enumerate(annotated):
            p=os.path.join(tmp,f"p{i:03d}.png"); img.save(p,"PNG"); paths.append(p)
        with open(pdf_out,"wb") as f: f.write(img2pdf.convert(paths))
    print(f"[+] → {pdf_out}")

if __name__=="__main__": main()
