CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                            C
C       ------ DSSV POLARIZED PARTON DISTRIBUTIONS ------                    C
C                                                                            C
C       REFERENCE:                                                           C
C           "GLOBAL ANALYSIS OF HELICITY PARTON DENSITIES                    C
C            AND THEIR UNCERTAINTIES",                                       C
C            arXiv:0804:0422 [hep-ph]                                        C
C                                                                            C
C       INPUT:                                                               C
C           X     :  BJORKEN-X BETWEEN 10**(-5)< X < 1.0                     C
C           Q2    :  SCALE**2 IN GeV**2 BETWEEN  1 < Q**2 < 10**5 GeV**2     C
C                                                                            C
C       OUTPUT:                                                              C
C           DUV   :  X * U VALENCE DISTRIBUTION                              C
C           DDV   :  X * D VALENCE DISTRIBUTION                              C
C           DUBAR :  X * UBAR DISTRIBUTION                                   C
C           DDBAR :  X * DBAR DISTRIBUTION                                   C
C           DSTR  :  X * STRANGE DISTRIBUTION                                C
C           DGLU  :  X * GLUON DISTRIBUTION                                  C
C                                                                            C
C       IMPORTANT:                                                           C
C           ALWAYS X*DISTRIBUTION IS RETURNED !!!                            C
C           ALL PDFs ARE IN THE MSbar SCHEME                                 C
C                                                                            C
C           BEFORE CALLING THE SUBROUTINE 'DSSVFIT' FOR THE FIRST TIME,      C
C           THE SUBROUTINE 'DSSVINI' MUST BE CALLED (ONLY ONCE) TO           C
C           INITIALIZE THE GRIDS !!                                          C
C                                                                            C
C       IN CASE OF PROBLEMS, DOUBTS, ETC, PLEASE E-MAIL US:                  C
C           D. de Florian  deflo@df.uba.ar                                   C
C           R. Sassot      sassot@df.uba.ar                                  C
C           M. Stratmann   marco@ribf.riken.jp                               C
C           W. Vogelsang   vogelsan@quark.phy.bnl.gov                        C
C                                                                            C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C---------------------------------------------------------------------
      SUBROUTINE DSSVFIT(X,Q2,DUV,DDV,DUBAR,DDBAR,DSTR,DGLU)
C---------------------------------------------------------------------
      IMPLICIT NONE
C...
      INTEGER NPART, NX, NQ, NARG
      PARAMETER (NPART=6, NX=47, NQ=30, NARG=2)
C...
      INTEGER IQ, IX
      INTEGER NA(NARG)
      DOUBLE PRECISION XUF(NX,NQ), XDF(NX,NQ), XUBF(NX,NQ), XDBF(NX,NQ), 
     1          XSF(NX,NQ), XGF(NX,NQ), XG1P(NX,NQ), XG1N(NX,NQ),
     2          PARTON (NPART,NQ,NX-1), QS(NQ), XB(NX), XT(NARG), 
     3          ARRF(NX+NQ) 
      DOUBLE PRECISION X, Q2
      DOUBLE PRECISION DUV, DDV, DUBAR, DDBAR, DSTR, DGLU
      DOUBLE PRECISION XB0, XB1
      DOUBLE PRECISION FFINT
C...      
      COMMON/ DSSVGRID / XUF, XDF, XUBF, XDBF, XSF, XGF, NA, ARRF
C
Cf2py intent(in) x 
Cf2py intent(in) q2 
Cf2py intent(out) duv 
Cf2py intent(out) ddv 
Cf2py intent(out) dubar 
Cf2py intent(out) ddbar 
Cf2py intent(out) dstr 
Cf2py intent(out) dglu 

C...  CHECK IF X AND Q2 VALUES ARE WITHIN RANGE OF THE GRID: 
C
      IF ( (X.LT.1.0D-5) .OR. (X.GT.1.0D0) ) THEN
         WRITE(6,91) 
 91      FORMAT (2X,'PARTON INTERPOLATION: X OUT OF RANGE')
         STOP 
      ENDIF
      IF ( (Q2.LT.0.8D0) .OR. (Q2.GT.1.D6) ) THEN
         WRITE(6,92) 
 92      FORMAT (2X,'PARTON INTERPOLATION: Q2 OUT OF RANGE')
         STOP
      ENDIF
C
C...  INTERPOLATION AND OUTPUT:
C
      XT(1) = DLOG(X)
      XT(2) = DLOG(Q2)
      DUV   = FFINT(NARG,XT,NA,ARRF,XUF)  * (1.D0-X)**3 * X
      DDV   = FFINT(NARG,XT,NA,ARRF,XDF)  * (1.D0-X)**4 * X
      DUBAR = FFINT(NARG,XT,NA,ARRF,XUBF) * (1.D0-X)**8 * X**0.5
      DDBAR = FFINT(NARG,XT,NA,ARRF,XDBF) * (1.D0-X)**8 * X**0.5
      DSTR  = FFINT(NARG,XT,NA,ARRF,XSF)  * (1.D0-X)**8 * X**0.5
      DGLU  = FFINT(NARG,XT,NA,ARRF,XGF)  * (1.D0-X)**5 * X**0.5
C...
 60   RETURN
      END
C
C-----------------------------
      SUBROUTINE DSSVINI(ISET)
C-----------------------------
      IMPLICIT NONE
C...
      INTEGER NPART, NX, NQ, NARG, ISET
      PARAMETER (NPART=6, NX=47, NQ=30, NARG=2)
C...
      INTEGER NA(NARG)
      DOUBLE PRECISION PARTON (NPART,NQ,NX-1)
      DOUBLE PRECISION QS(NQ), XB(NX)
      DOUBLE PRECISION XUF(NX,NQ), XDF(NX,NQ), XUBF(NX,NQ), XDBF(NX,NQ), 
     1          XSF(NX,NQ), XGF(NX,NQ), XT(NARG), ARRF(NX+NQ) 
      DOUBLE PRECISION XB0, XB1
      INTEGER M, N
      INTEGER IQ, IX

      character *100 fname1(2)
      character *2  fname2, fname2a
      character *5  fname3
      character *100 fname,file
      character *1 dummy0
C...
      DATA fname1 / 'analysis/qpdlib/sets/DSSV/Grids/dssv-plus-',
     1              'analysis/qpdlib/sets/DSSV/Grids/dssv-minus-'/
C...
      COMMON/ DSSVGRID / XUF, XDF, XUBF, XDBF, XSF, XGF, NA, ARRF
C
C...  BJORKEN-X AND Q**2 VALUES OF THE GRID :
C
      DATA QS / 0.8D0, 1.0D0, 1.25d0, 1.5D0, 2.d0, 2.5D0, 
     1     4.0D0, 6.4D0, 1.0D1, 1.5D1, 2.5D1, 4.0D1, 6.4D1,
     2     1.0D2, 1.8D2, 3.2D2, 5.8D2, 1.0D3, 1.8D3,
     3     3.2D3, 5.8D3, 1.0D4, 1.8D4, 3.2D4, 5.8D4, 
     4     1.0D5, 1.8D5, 3.2D5, 5.8D5, 1.0D6  /
      DATA XB / 
     1           1.0D-5, 4.D-5, 6.7D-5, 1.0D-4, 1.4D-4, 2.0D-4,
     2           3.0D-4, 4.5D-4, 6.7D-4, 1.0D-3, 1.4D-3, 2.0D-3,
     3           3.0D-3, 4.5D-3, 6.7D-3, 1.0D-2, 1.4D-2, 2.0D-2,
     4           3.0D-2, 4.5D-2, 0.06, 0.08, 0.1, 0.125, 0.15,
     5           0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325,
     6           0.35, 0.375, 0.4,  0.45, 0.5, 0.55, 0.6,
     7           0.65,  0.7,  0.75,  0.8,  0.85, 0.9, 0.95, 1.0/ 

C...
      fname3='.grid'
      dummy0='0'

      if(iset.eq.0) then
         file='analysis/qpdlib/sets/DSSV/Grids/DSSV-GRID.NLO'
C         file='DSSV-p2pc-allx.GRID'
      else if((iset.lt.0).and.(iset.ge.-9)) then
         write (fname2,1000) abs(iset)
 1000    format(I1)
         call strcat(dummy0,fname2,fname2a)
         call strcat(fname1(2),fname2a,fname)
         call strcat(fname,fname3,file)
      else if(iset.lt.-9) then
         write (fname2,1001) abs(iset)
 1001    format(I2)
         call strcat(fname1(2),fname2,fname)
         call strcat(fname,fname3,file)
      else if((iset.gt.0).and.(iset.le.9)) then
         write (fname2,1002) abs(iset)
 1002    format(I1)
         call strcat(dummy0,fname2,fname2a)
         call strcat(fname1(1),fname2a,fname)
         call strcat(fname,fname3,file)
      else if(iset.gt.9) then
         write (fname2,1003) abs(iset)
 1003    format(I2)
         call strcat(fname1(1),fname2,fname)
         call strcat(fname,fname3,file)
      endif

      OPEN(UNIT=71,FILE=file,STATUS='OLD')
C...
      DO 15 M = 1, NX-1
         DO 15 N = 1, NQ
            READ(71,*) PARTON(1,N,M), PARTON(2,N,M), PARTON(3,N,M), 
     1           PARTON(4,N,M), PARTON(5,N,M), PARTON(6,N,M)
 90         FORMAT (6(1PE12.4))
 15   CONTINUE
C...
      CLOSE(71)
C
C.... ARRAYS FOR THE INTERPOLATION SUBROUTINE :
C
      DO 10 IQ = 1, NQ
         DO 20 IX = 1, NX-1
            XB0 = XB(IX) 
            XB1 = 1.D0-XB(IX)
            XUF(IX,IQ)  = PARTON(1,IQ,IX) / (XB1**3 * XB0)
            XDF(IX,IQ)  = PARTON(2,IQ,IX) / (XB1**4 * XB0)
            XUBF(IX,IQ) = PARTON(3,IQ,IX) / (XB1**8 * XB0**0.5) 
            XDBF(IX,IQ) = PARTON(4,IQ,IX) / (XB1**8 * XB0**0.5) 
            XSF(IX,IQ)  = PARTON(5,IQ,IX) / (XB1**8 * XB0**0.5) 
            XGF(IX,IQ)  = PARTON(6,IQ,IX) / (XB1**5 * XB0**0.5)
 20      CONTINUE
         XUF(NX,IQ)  = 0.D0
         XDF(NX,IQ)  = 0.D0
         XUBF(NX,IQ) = 0.D0
         XDBF(NX,IQ) = 0.D0
         XSF(NX,IQ)  = 0.D0
         XGF(NX,IQ)  = 0.D0
 10   CONTINUE  
      NA(1) = NX
      NA(2) = NQ
      DO 30 IX = 1, NX
         ARRF(IX) = DLOG(XB(IX))
 30   CONTINUE
      DO 40 IQ = 1, NQ
         ARRF(NX+IQ) = DLOG(QS(IQ))
 40   CONTINUE
 16   CONTINUE
C...
      RETURN
      END
*
*...CERN LIBRARY ROUTINE E104 (INTERPOLATION) :
*
      FUNCTION FFINT(NARG,ARG,NENT,ENT,TABLE)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION ARG(5),NENT(5),ENT(73),TABLE(1200)
      DIMENSION D(5),NCOMB(5),IENT(5)
      KD=1
      M=1
      JA=1
      DO 5 I=1,NARG
         NCOMB(I)=1
         JB=JA-1+NENT(I)
         DO 2 J=JA,JB
            IF (ARG(I).LE.ENT(J)) GO TO 3
 2       CONTINUE
         J=JB
 3       IF (J.NE.JA) GO TO 4
         J=J+1
 4       JR=J-1
         D(I)=(ENT(J)-ARG(I))/(ENT(J)-ENT(JR))
         IENT(I)=J-JA
         KD=KD+IENT(I)*M
         M=M*NENT(I)
 5    JA=JB+1
      FFINT=0.D0
 10   FAC=1.D0
      IADR=KD
      IFADR=1
      DO 15 I=1,NARG
         IF (NCOMB(I).EQ.0) GO TO 12
         FAC=FAC*(1.D0-D(I))
         GO TO 15
 12      FAC=FAC*D(I)
         IADR=IADR-IFADR
 15   IFADR=IFADR*NENT(I)
      FFINT=FFINT+FAC*TABLE(IADR)
      IL=NARG
 40   IF (NCOMB(IL).EQ.0) GO TO 80
      NCOMB(IL)=0
      IF (IL.EQ.NARG) GO TO 10
      IL=IL+1
      DO 50  K=IL,NARG
 50   NCOMB(K)=1
      GO TO 10
 80   IL=IL-1
      IF(IL.NE.0) GO TO 40
      RETURN
      END

c----------------------------------------------------------------------------------------
      subroutine strcat(str1,str2,str)
c concatenates str1 and str2 into str. Ignores trailing blanks of str1,str2
      character *(*) str1,str2,str
      l1=istrl(str1)
      l2=istrl(str2)
      l =len(str)
      if(l.lt.l1+l2) then
          write(*,*) 'error: l1+l2>l in strcat'
          write(*,*) 'l1=',l1,' str1=',str1
          write(*,*) 'l2=',l2,' str2=',str2
          write(*,*) 'l=',l
          stop
      endif
      if(l1.ne.0) str(1:l1)=str1(1:l1)
      if(l2.ne.0) str(l1+1:l1+l2)=str2(1:l2)
      if(l1+l2+1.le.l) str(l1+l2+1:l)= ' '
      end
c
      function istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      istrl = i
      end
c----------------------------------------------------------------------------------------
