import numpy as np
import matplotlib.pyplot as plt
 
# ── MATERIAL: IM7/8552 (matches eLamX2 fields exactly) ──
E1  = 171000.0    # eLamX2: E||
E2  = 9080.0      # eLamX2: E_perp
G12 = 5290.0      # eLamX2: G||_perp
v12 = 0.32        # eLamX2: v||_perp
t_ply = 0.131     # eLamX2: set per-ply in Laminate editor
Xt  = 2326.0      # eLamX2: Xt
Xc  = 1200.0      # eLamX2: Xc
Yt  = 62.3        # eLamX2: Yt
Yc  = 199.8       # eLamX2: Yc
S12 = 92.3        # eLamX2: SC
v21 = v12 * E2 / E1
 
denom = 1 - v12 * v21
Q = np.array([
    [E1/denom,       v12*E2/denom, 0   ],
    [v12*E2/denom,   E2/denom,     0   ],
    [0,              0,            G12 ]])
 
def T_mat(a):
    t = np.radians(a)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c**2,s**2,2*c*s],[s**2,c**2,-2*c*s],[-c*s,c*s,c**2-s**2]])
 
def Qbar(a):
    T = T_mat(a); Ti = np.linalg.inv(T)
    R = np.diag([1,1,2]); Ri = np.diag([1,1,0.5])
    return Ti @ Q @ R @ T @ Ri
 
def ABD(angles):
    n = len(angles); h = n * t_ply
    z = np.linspace(-h/2, h/2, n+1)
    A = B = D = np.zeros((3,3))
    for k in range(n):
        Qb = Qbar(angles[k])
        A = A + Qb*(z[k+1]-z[k])
        B = B + 0.5*Qb*(z[k+1]**2-z[k]**2)
        D = D + (1/3)*Qb*(z[k+1]**3-z[k]**3)
    return A, B, D
 
def tsai_wu(s1,s2,t12):
    F1=1/Xt-1/Xc; F2=1/Yt-1/Yc
    F11=1/(Xt*Xc); F22=1/(Yt*Yc); F66=1/S12**2
    F12=-0.5*np.sqrt(F11*F22)  # matches eLamX2 F12*=-0.5
    return F1*s1+F2*s2+F11*s1**2+F22*s2**2+F66*t12**2+2*F12*s1*s2
 
layups = {
    "Cross-Ply [0/90]s":         [0,90,90,0],
    "Quasi-Iso [0/+45/-45/90]s": [0,45,-45,90,90,-45,45,0],
    "Tailored [0_2/pm45/0]s":    [0,0,45,-45,0,0,-45,45,0,0],
}
 
print("="*72)
print("  CFRP LAMINATE ANALYSIS - IM7/8552 - ply t=0.131mm")
print("="*72)
 
for name, angles in layups.items():
    A,_,D = ABD(angles)
    h = len(angles)*t_ply
    Ex=(A[0,0]*A[1,1]-A[0,1]**2)/(A[1,1]*h)
    Ey=(A[0,0]*A[1,1]-A[0,1]**2)/(A[0,0]*h)
    Gxy=A[2,2]/h; vxy=A[0,1]/A[1,1]
    print(f"\n{'='*72}\n  {name}  |  {len(angles)} plies  |  {h:.3f} mm")
    print(f"  Ex={Ex:,.0f}  Ey={Ey:,.0f}  Gxy={Gxy:,.0f}  vxy={vxy:.4f}")
    print(f"  A11={A[0,0]:.1f}  A22={A[1,1]:.1f}  A66={A[2,2]:.1f}")
    print(f"  D11={D[0,0]:.4f}  D22={D[1,1]:.4f}  D66={D[2,2]:.4f}")
    for tag,Nv in [("Nx=1000",[1000,0,0]),("Nxy=500",[0,0,500])]:
        eps=np.linalg.solve(A,np.array(Nv,dtype=float))
        print(f"\n  Tsai-Wu | {tag}:")
        print(f"  {'Ply':>4} {'Ang':>5} {'sig1':>9} {'sig2':>9}"
              f" {'tau12':>9} {'FI':>8} {'RF':>8} {'OK?':>5}")
        for i,a in enumerate(angles):
            sl = T_mat(a) @ (Qbar(a) @ eps)
            fi = tsai_wu(sl[0],sl[1],sl[2])
            rf = 1/fi if fi>0 else 999
            print(f"  {i+1:4d} {a:+5.0f} {sl[0]:9.1f} {sl[1]:9.1f}"
                  f" {sl[2]:9.1f} {fi:8.4f} {rf:8.2f}"
                  f" {'SAFE' if rf>1 else 'FAIL':>5}")
 
# ── POLAR PLOTS ──
fig,axes=plt.subplots(1,3,figsize=(16,5.5),subplot_kw={"projection":"polar"})
fig.suptitle("Laminate Stiffness vs Direction",fontsize=14,fontweight="bold",y=1.02)
colors=["#C0392B","#0078D7","#27AE60"]
for idx,(name,angles) in enumerate(layups.items()):
    A,_,_=ABD(angles); h=len(angles)*t_ply
    th=np.linspace(0,2*np.pi,360); vals=[]
    for t in th:
        Ti=np.linalg.inv(T_mat(np.degrees(t)))
        Ar=Ti@A@Ti.T
        vals.append(max((Ar[0,0]*Ar[1,1]-Ar[0,1]**2)/(Ar[1,1]*h),0))
    axes[idx].plot(th,vals,color=colors[idx],lw=2)
    axes[idx].fill(th,vals,alpha=0.12,color=colors[idx])
    axes[idx].set_title(name,fontsize=9,pad=18,fontweight="bold")
    axes[idx].set_rticks([])
plt.tight_layout()
plt.savefig("polar_plots.png",dpi=150,bbox_inches="tight")
print("\n>>> Saved polar_plots.png")
plt.show()