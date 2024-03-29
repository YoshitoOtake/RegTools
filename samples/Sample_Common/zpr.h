#ifndef ZPR_H
#define ZPR_H

/*
 * Zoom-pan-rotate mouse manipulation module for GLUT
 * Version 0.4, October 2003
 *
 * Nigel Stewart
 * School of Computer Science and Information Technology
 * RMIT University
 * nigels@cs.rmit.edu.au
 *
 * Instructions
 * ------------
 *
 * Call zprInit() immediately after your call to glutCreateWindow()
 *
 * The ZPR module handles glutReshapeFunc(), glutMouseFunc() and glutMotionFunc()
 * Applications should not bypass the ZPR handlers for reshape or mouse events.
 *
 * Mouse manipulation of the GLUT window via the modelview matrix:
 *
 * Left   button -> rotate
 * Middle button -> zoom
 * Right  button -> pan
 *
 * Picking is also provided via two configurable callbacks:
 *
 * void zprSelectionFunc(void (*f)(void))
 *
 *   The draw function to be called in OpenGL selection
 *   mode in response to a mouse-down button event.
 *
 * void zprPickFunc(void (*f)(GLint name))
 *
 *   The callback function which will receive the
 *   top-most item of the name stack of the closest selection
 *   hit.  If there is no selection hit, -1
 *
 * Limitations
 * -----------
 *
 * Works best with zprReferencePoint appropriately configured.
 * Works best with ortho projection.
 * You may need to use glEnable(GL_NORMALIZATION) for correct lighting.
 * Near and far clip planes are hard-coded.
 * Zooming and rotation is centered on the origin.
 * Only one window can use the callbacks at one time.
 *
 */

#ifdef WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/*
 *
 */

/* Mouse Manipulation API */

void zprInit();

extern GLfloat zprReferencePoint[4];
extern float intensity_window_current;
extern float intensity_level_current;
extern float intensity_addjustment_scaling;
extern float ray_casting_thresh;
extern void UpdateWindowLevel();
extern int isRayCasting;

/* Picking API (Optional) */

extern void zprSelectionFunc(void (*f)(void));      /* Selection-mode draw function */
extern void zprPickFunc(void (*f)(GLint name));     /* Pick event handling function */

/*
 *
 */

#ifdef __cplusplus
}
#endif

#endif
