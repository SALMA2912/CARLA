import carla
import numpy as np
import cv2
import time
import math
import threading
import random
import traceback
from typing import Tuple, Dict

#*********************************COORDINATE SYSTEM*********************************
#***********************************************************************************
# +X : forward
# +Y : right
# yaw: 0 deg along +X, positive is counter-clockwise
# ex : longitudinal target error in ego frame (target is ahead)
# ey : lateral target error in ego frame

#*********************************CONFIGURATION CONSTANTS*********************************
#*****************************************************************************************
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SIMULATION CONSTANTS////////////////////////////////////

HOST  = "localhost"
PORT  = 2000
TOWN  = "Town05"
FINAL_HOLD_SECONDS = 15.0       # how long to keep the cars visible after parking
DEBUG_EVERY_N  = 60
WIN_W, WIN_H   = 640, 480


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SPAWNED EGO VEHICLE//////////////////////////////////// 

SPAWN_X   = -50.0
SPAWN_Y   =   4.5
SPAWN_Z   =   0.3  
SPAWN_YAW =   0.0

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\PARKED VEHICLES//////////////////////////////////// 

PARK_CAR_A_BP    = "vehicle.tesla.model3"      # front blocker car
PARK_CAR_B_BP    = "vehicle.audi.tt"           # rear blocker car

PARK_CAR_A_COLOR = "255,255,255"               # white
PARK_CAR_B_COLOR = "50,50,50"                  # dark grey

PARK_CAR_A_X, PARK_CAR_A_Y, PARK_CAR_A_Z, PARK_CAR_A_YAW = 3.0,  7, 0.3, 0.0 
PARK_CAR_B_X, PARK_CAR_B_Y, PARK_CAR_B_Z, PARK_CAR_B_YAW = 15.0,  7, 0.3, 0.0


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\MOTION TUNING////////////////////////////////////

                                
MIN_SLOT_LENGTH   = 6.2         #min gap that counts as valid parking slot
VEHICLE_LENGTH    = 4.7


#Speed (km/h)
SCAN_SPEED_KMH     = 4.5       #cruise speed during slot scanning and forward align
MANEUVER_SPEED_KMH = 2.2       #max speed during reverse

#Final pose tolerances
DIST_FWD_CTR = 3.2             #maximum forward travel during the centering phase 
TARGET_EX_TOL = 0.35           #longitudinal tolerance for parking completion. 
TARGET_EY_TOL = 0.40           #lateral tolerance for parking completion.
TARGET_YAW_TOL = 4.0           #heading tolerance for parking completion.

#Slot tuning
PARKING_DEPTH_BIAS =1.75       #push car deeper into the slot so the car does not stop outside it
ALIGN_PASS_EXTRA = 2.4         #pass the slot farther before reversing to improve entry arc 

#Centering tuning
LATERAL_RECOVERY_EY = 1.7      #if ey exceeds this during centering, reverse-into-slot instead of trying to drive forward. 
CENTER_DIR_SWITCH_COOLDOWN = 18 #ticks to wait before changing centering drive direction
CENTER_STAGE3_MAX_TRAVEL = 0.2  
CENTER_STAGE3_EY_DONE = 0.25

#Reverse phase switching
REV_PHASE1_TRIGGER_EY = 4      #lateral error that ends Phase 1 and starts Phase 2
REV_PHASE1_OVERSHOOT_EX = -6   #longitudinal overshoot that ends Phase 1, preventing the car driving too deep 
REVERSE_FINISH_EY = 0.35       #lateral tolerance to exit Phase 2 
REVERSE_FINISH_EX = 0.35       #longitudinal tolerance to exit Phase 2

REV1_MAX_TRAVEL = 6.0    
REV1_TARGET_YAW = 50.0       
       

#Stall detection
STALL_SPEED_KMH = 0.15         #speed below which the stall counter increments.
STALL_TICKS_LIMIT = 45         #~2.25 s at 20 fps

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\LiDAR CONFIGURATION////////////////////////////////////
LIDAR_RANGE    = 20.0          #max detection range
LIDAR_CHANNELS = 32            #number of vertical scan lines
LIDAR_PPS      = 120_000       #horizontal density per revolution
LIDAR_FPS      = 20.0          #rotation frequency

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\LiDAR TUNING////////////////////////////////////


EGO_HALF_LENGTH = 2.45
EGO_HALF_WIDTH  = 1.05

SAFETY_POINT_Z_MIN = -1.6
SAFETY_POINT_Z_MAX = 1.4
SAFETY_MIN_POINTS = 3

SAFETY_FORWARD_LOOKAHEAD = 4.0
SAFETY_REVERSE_LOOKAHEAD = 4.0
SAFETY_BASE_SIDE_MARGIN = 0.55
SAFETY_STEER_SIDE_GAIN = 1.35   
SAFETY_CROSS_MARGIN = 0.35      #keep some margin on the opposite side 

SELF_FILTER_X_MARGIN = 0.20     #extra margin to reject LiDAR hits from ego body/mount
SELF_FILTER_Y_MARGIN = 0.20     #extra margin to reject LiDAR hits from ego body/mount

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\MAP LAYERS TO UNLOAD////////////////////////////////////


UNLOAD_LAYERS = [
    carla.MapLayer.Buildings,
    carla.MapLayer.Foliage,
    carla.MapLayer.ParkedVehicles,
    carla.MapLayer.Walls,
    carla.MapLayer.Decals,
    carla.MapLayer.Props,
    carla.MapLayer.StreetLights,
]


#*********************************HELPER FUNCTIONS*********************************
#**********************************************************************************
#Limits the value v to the range used for throttle and steer
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


#Converts to m/s
def speed_kmh(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


#Measure how far the car has moved since the last state transition.
def dist2d(a: carla.Location, b: carla.Location):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


#Ensures heading errors do not jump discontinuously near ±180°.
def normalize_angle_deg(a):
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


#Converts a point in the ego vehicle's local frame  into world coordinates
def world_from_ego(vehicle, x_local, y_local):
    tf = vehicle.get_transform()
    loc = tf.location
    yaw = math.radians(tf.rotation.yaw)

    wx = loc.x + x_local * math.cos(yaw) - y_local * math.sin(yaw)
    wy = loc.y + x_local * math.sin(yaw) + y_local * math.cos(yaw)
    return carla.Location(x=wx, y=wy, z=loc.z)


#Applies clamp to all values, and sends it to the vehicle
def apply_ctrl(vehicle, throttle=0.0, steer=0.0, brake=0.0, reverse=False):
    c            = carla.VehicleControl()
    c.throttle   = clamp(float(throttle), 0.0, 1.0)
    c.steer      = clamp(float(steer),   -1.0, 1.0)
    c.brake      = clamp(float(brake),    0.0, 1.0)
    c.reverse    = reverse
    c.hand_brake = False
    vehicle.apply_control(c)



#*********************************SPEED CONTROLLER CLASS*********************************
#****************************************************************************************
class SpeedPID:
    def __init__(self, kp=0.15, min_thr=0.35, max_thr=0.70):
        self.kp      = kp
        self.min_thr = min_thr #prevents stalling
        self.max_thr = max_thr  #prevents excessive acc


    def step(self, current_kmh: float, target_kmh: float) -> float:
        error = target_kmh - current_kmh
        if error <= 0:
            return 0.0                        
        raw = self.kp * error + self.min_thr  
        return clamp(raw, self.min_thr, self.max_thr)


#**********************************MAP CLEANUP FUNCTION*********************************
#***************************************************************************************
def strip_map(world: carla.World):
    print("\nUnloading map layers …")
    for layer in UNLOAD_LAYERS:
        try:
            world.unload_map_layer(layer)
            print(f"  ✓  {layer}")
        except Exception as e:
            print(f"  ✗  {layer}: {e}")

    removed = 0
    for actor in world.get_actors().filter("vehicle.*"):
        vel = actor.get_velocity()
        if math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) < 0.01:
            try: 
                actor.destroy() 
                removed += 1
            except Exception: 
                pass
    print(f"  ✓  {removed} static vehicles removed\n")



#*********************************SENSOR BRIDGE CLASS*********************************
#*************************************************************************************
#Manages thread-safe access to sensor data
class SensorBridge:
    def __init__(self):
        self._lock       = threading.Lock()
        self._lidar      = np.empty((0, 4), dtype=np.float32)
        self._seg        = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self.lidar_count = 0
    

    def on_lidar(self, data: carla.LidarMeasurement):
        raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        with self._lock:
            self._lidar = raw.copy() 
            self.lidar_count += 1
    

    def on_seg(self, image: carla.Image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr = cv2.resize(arr[:, :, :3][:, :, ::-1], (WIN_W, WIN_H))
        with self._lock:
            self._seg = bgr
    

    def lidar(self):
        with self._lock: 
            return self._lidar.copy(), self.lidar_count
    

    def seg(self):
        with self._lock: 
            return self._seg.copy()



#*********************************SLOT DETECTOR CLASS*********************************
#*************************************************************************************
#Processes the raw LiDAR point cloud each tick to find the empty parking gap
class SlotDetector:
    def __init__(self, min_length=MIN_SLOT_LENGTH,  side='right'):
        self.min_length = min_length
        self.side       = side


    def detect(self, pts: np.ndarray):
        if pts.shape[0] < 10:
            return None

        chosen =self._filter(pts, self.side)
        if chosen.shape[0] < 5:
            return None
        return self._find_gap(chosen)


    def debug_info(self, pts):
        if pts.shape[0] == 0: 
            return "LiDAR: 0 pts"
        r = self._filter(pts, 'right').shape[0]
        l = self._filter(pts, 'left').shape[0]
        return (
            f"total={pts.shape[0]}  R={r}  L={l}  "
            f"X:[{pts[:,0].min():.1f},{pts[:,0].max():.1f}]  "
            f"Y:[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]"
            )
    

#Keeps only points that lie on the correct side and within a sensible distance range
    def _filter(self, pts, side):
        if side == "right":
            y_mask = (pts[:, 1] > 0.5) & (pts[:, 1] < 7.0)
        else:
            y_mask = (pts[:, 1] < -0.5) & (pts[:, 1] > -7.0)
       
        xyz_mask = (
            (pts[:, 2] > -1.5) & (pts[:, 2] < 1.5) &
            (pts[:, 0] > -2.0) & (pts[:, 0] < 30.0)
        )

        return pts[y_mask & xyz_mask]


#A bin is considered occupied if it contains at least 3 points at a mean lateral distance below 6 m
    def _find_gap(self, side_pts):
        xs = side_pts[:, 0]
        ys = np.abs(side_pts[:, 1])
        BIN  = 0.5
        x_lo = max(0.0, float(xs.min()))
        x_hi = float(xs.max())
        print(f"[GAP] x_lo={x_lo:.2f}, x_hi={x_hi:.2f}, span={x_hi-x_lo:.2f}, pts={len(xs)}")

        if x_hi - x_lo < self.min_length:
            return None
        
        bins     = np.arange(x_lo, x_hi + BIN, BIN)
        occ_bins = []
        for i in range(len(bins) - 1):
            in_b = (xs >= bins[i]) & (xs < bins[i+1])
            if in_b.sum() >= 3 and float(ys[in_b].mean()) < 6.0:
                occ_bins.append(float(bins[i]))

        if len(occ_bins) < 2:
            print("[GAP] Rejected: not enough occupied bins")
            return None
        
        occ  = np.array(occ_bins)
        gaps = np.diff(occ)

        print(f"[GAP] gaps={gaps}")
        best = int(np.argmax(gaps))

        if float(gaps[best]) < self.min_length:
            print(f"[GAP] Rejected: best gap {gaps[best]:.2f} < min_length {self.min_length:.2f}")
            return None
        
        sx   = float(occ[best])
        ex   = float(occ[best+1])
        near = (xs > sx - 2.0) & (xs < sx + 2.0)
        yd   = float(ys[near].mean()) if near.sum() > 0 else 3.0

        print(f"[GAP] SLOT FOUND: start={sx:.2f}, end={ex:.2f}, y_dist={yd:.2f}")

        return {"slot_start_x": sx, "slot_end_x": ex,
                "slot_y_dist": yd, "side": self.side}




#*********************************LiDAR SAFETY MONITOR CLASS*********************************
#********************************************************************************************
class LidarSafetyMonitor:
    def __init__(self):
        self.last = {
            "any": {"clearance": -1.0, "points": 0},
            "forward": {"clearance": -1.0, "points": 0},
            "reverse": {"clearance": -1.0, "points": 0},
        }


    def _prepare_points(self, pts: np.ndarray) -> np.ndarray:
        if pts is None or pts.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        finite_and_height = (
            np.isfinite(pts[:, 0]) & 
            np.isfinite(pts[:, 1]) & 
            np.isfinite(pts[:, 2]) &
            (pts[:, 2] > SAFETY_POINT_Z_MIN) & 
            (pts[:, 2] < SAFETY_POINT_Z_MAX)
        )

        pts_xyz = pts[finite_and_height][:, :3]
        if pts_xyz.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        outside_ego = (
            (np.abs(pts_xyz[:, 0]) > (EGO_HALF_LENGTH + SELF_FILTER_X_MARGIN)) |
            (np.abs(pts_xyz[:, 1]) > (EGO_HALF_WIDTH  + SELF_FILTER_Y_MARGIN))
        )
        pts_xyz = pts_xyz[outside_ego]
        return pts_xyz.astype(np.float32, copy=False)


    def _body_clearance(self, pts_xyz: np.ndarray) -> np.ndarray:
        dx = np.maximum(np.abs(pts_xyz[:, 0]) - EGO_HALF_LENGTH, 0.0)
        dy = np.maximum(np.abs(pts_xyz[:, 1]) - EGO_HALF_WIDTH, 0.0)
        return np.sqrt(dx * dx + dy * dy)


    def _sector_mask(self, pts_xyz: np.ndarray, direction: str, steer: float) -> np.ndarray:
        if pts_xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        steer = clamp(float(steer), -1.0, 1.0)
        turning_right = steer > 0.05
        turning_left = steer < -0.05

        if direction == "forward":
            x_min = -0.8
            x_max = EGO_HALF_LENGTH + SAFETY_FORWARD_LOOKAHEAD
            primary = pts_xyz[:, 0] >= x_min
        else:
            x_min = -(EGO_HALF_LENGTH + SAFETY_REVERSE_LOOKAHEAD)
            x_max = 0.8
            primary = pts_xyz[:, 0] <= x_max

        extra_right = SAFETY_BASE_SIDE_MARGIN + (SAFETY_STEER_SIDE_GAIN * abs(steer) if turning_right else SAFETY_CROSS_MARGIN)
        extra_left  = SAFETY_BASE_SIDE_MARGIN + (SAFETY_STEER_SIDE_GAIN * abs(steer) if turning_left  else SAFETY_CROSS_MARGIN)

        y_min = -(EGO_HALF_WIDTH + extra_left)
        y_max = +(EGO_HALF_WIDTH + extra_right)

        mask = (primary & 
                (pts_xyz[:, 0] >= x_min) & 
                (pts_xyz[:, 0] <= x_max) & 
                (pts_xyz[:, 1] >= y_min) & 
                (pts_xyz[:, 1] <= y_max))
        return mask


    def clearance(self, pts: np.ndarray, direction: str, steer: float) -> dict:
        pts_xyz = self._prepare_points(pts)
        if pts_xyz.shape[0] == 0:
            out = {"clearance": -1.0, "points": 0}
            self.last[direction] = out
            return out

        mask = self._sector_mask(pts_xyz, direction, steer)
        cand = pts_xyz[mask]
        if cand.shape[0] == 0:
            out = {"clearance": -1.0, "points": 0}
            self.last[direction] = out
            return out

        clr = self._body_clearance(cand)
        order = np.argsort(clr)
        keep_n = min(max(SAFETY_MIN_POINTS, 1), clr.shape[0])
        robust = float(np.median(clr[order[:keep_n]]))
        out = {"clearance": robust, "points": int(cand.shape[0])}
        self.last[direction] = out
        return out


    def should_stop(self, pts: np.ndarray, reverse: bool, steer: float, speed_kmh_now: float) -> Tuple[bool, Dict]:
        direction = "reverse" if reverse else "forward"
        info = self.clearance(pts, direction, steer)
        if info["points"] <= 0 or info["clearance"] < 0.0:
            return False, info

        speed_ms = max(0.0, speed_kmh_now / 3.6)
        if direction == "reverse":
            threshold = 0.40 + 0.22 * speed_ms
            hard_stop = 0.18
        else:
            threshold = 0.34 + 0.16 * speed_ms
            hard_stop = 0.22

        info["threshold"] = threshold
        info["hard_stop"] = hard_stop
        info["direction"] = direction

        if info["clearance"] <= hard_stop:
            info["stop_reason"] = "hard_stop"
            return True, info

        if info["clearance"] <= threshold:
            info["stop_reason"] = "soft_stop"
            return True, info

        return False, info



#*********************************PARKING FINITE STATE MACHINE*********************************
#**********************************************************************************************
#STATE FLOW:
#SCANNING-> FWD_ALIGN-> STOP_ALIGN-> REVERSE_R-> REVERSE_L-> FWD_CTR-> PARKED
#SCANNING   : Drive forward at scan speed, LiDAR is continuously processed. Exits when a valid parking gap is found
#FWD_ALIGN  : drive forward until we have passed slot_start  by ALIGN_PASS_EXTRA X VEHICLE_LENGTH
#STOP_ALIGN : full brake until speed < 0.5 km/h
#REVERSE_R  : reverse + full steer away from slot until DIST_REV_R metres travelled
#REVERSE_L  : reverse + full steer toward slot until DIST_REV_L metres travelled
#FWD_CTR    : forward straight until DIST_FWD_CTR metres travelled
#PARKED     : hold brake

class ParkingFSM:
    SCANNING = "SCANNING"
    FWD_ALIGN = "FORWARD_ALIGN"
    STOP = "STOP_ALIGN"
    REV_RIGHT = "REVERSE_RIGHT"
    REV_LEFT = "REVERSE_LEFT"
    FWD_CTR = "FORWARD_CENTER"
    PARKED = "PARKED"

    def __init__(self, vehicle, detector):
        self.vehicle = vehicle
        self.detector = detector

        self.state = self.SCANNING
        self.slot = None
        self._park_side = "right"

        self._ref = vehicle.get_location() 
        self._yaw_ref = 0.0 
        self._tick = 0 

        self._pid = SpeedPID(kp=0.15, min_thr=0.38, max_thr=0.65)
        self._safety = LidarSafetyMonitor()  

        self._last_safety_msg_tick = -10_000
        self._stall_ticks = 0

        self.last_safety_stop = None  
        self.safety_stop_counter = 0 

        self._center_mode = None 
        self._center_last_switch_tick = -10_000 

        
        self._fwd_ctr_stage = "SET_STEER"
        self._fwd_ctr_stage_tick = 0

        self._log("Ready → SCANNING")


    def _log(self, m):
        print(f"  [{self.state:>22s}]  {m}")


    def _travel(self):
        return dist2d(self.vehicle.get_location(), self._ref)


    def _go(self, new_state, note=""):
        self.state = new_state
        self._ref = self.vehicle.get_location()
        self._yaw_ref = self.vehicle.get_transform().rotation.yaw
        self._log(note)


#Computes the signed angular difference between target_yaw and the car's current yaw
    def _heading_error(self, target_yaw):
        yaw = self.vehicle.get_transform().rotation.yaw
        return normalize_angle_deg(target_yaw - yaw)
    

#Projects the vector from the car to the world target point into the car's own local frame
    def _target_error_local(self, target_world):
        tf = self.vehicle.get_transform()
        loc = tf.location
        yaw = math.radians(tf.rotation.yaw)

        dx = target_world.x - loc.x
        dy = target_world.y - loc.y

        ex = dx * math.cos(yaw) + dy * math.sin(yaw)
        ey = -dx * math.sin(yaw) + dy * math.cos(yaw)  
        return ex, ey
    

#Converts the LiDAR-relative slot centre into a fixed world coordinate 
    def _slot_pose_from_detection(self, result):
        slot_center_x = 0.5 * (result["slot_start_x"] + result["slot_end_x"])

        sign = 1.0 if result["side"] == "right" else -1.0
        slot_center_y = sign * (result["slot_y_dist"] + PARKING_DEPTH_BIAS)

        target_world = world_from_ego(self.vehicle, slot_center_x, slot_center_y)
        target_yaw = self.vehicle.get_transform().rotation.yaw

        self.slot = dict(result)
        self.slot["slot_center_x"] = slot_center_x
        self.slot["slot_center_y"] = slot_center_y
        self.slot["target_world"] = target_world
        self.slot["target_yaw"] = target_yaw


    def _update_stall_counter(self, spd):
        if spd < STALL_SPEED_KMH:
            self._stall_ticks += 1
        else:
            self._stall_ticks = 0

    def _set_center_mode(self, mode):
        if mode != self._center_mode:
            self._center_mode = mode
            self._center_last_switch_tick = self._tick


    def _can_switch_center_mode(self):
        return (self._tick - self._center_last_switch_tick) >= CENTER_DIR_SWITCH_COOLDOWN


    def _unstick_reverse_into_slot(self, ey, heading_err, spd):
        steer_sign = 1.0 if ey > 0.0 else -1.0
        steer = clamp(0.88 * steer_sign + 0.018 * heading_err, -1.0, 1.0)
        thr = max(0.42, self._pid.step(spd, 2.5))
        apply_ctrl(self.vehicle, throttle=thr, steer=steer, reverse=True)


    def _lidar_safety_enabled(self):
        return self.state in {self.STOP, self.REV_RIGHT, self.REV_LEFT, self.FWD_CTR, self.PARKED}


    def _safe_apply_ctrl(self, pts, throttle=0.0, steer=0.0, brake=0.0, reverse=False):
        if brake >= 0.95:
            self.last_safety_stop = None
            self.safety_stop_counter = 0
            apply_ctrl(self.vehicle, throttle=0.0, steer=steer, brake=brake, reverse=reverse)
            return True
        
        if self._lidar_safety_enabled():
            should_stop, info = self._safety.should_stop(
                pts,
                reverse=reverse,
                steer=steer,
                speed_kmh_now=speed_kmh(self.vehicle),
            )
            if should_stop:
                self.last_safety_stop = dict(info)
                self.safety_stop_counter += 1
                apply_ctrl(self.vehicle, throttle=0.0, steer=0.0, brake=1.0, reverse=reverse)
                if (self._tick - self._last_safety_msg_tick) >= 8:
                    print(
                        f"  [SAFETY STOP] state={self.state}  {info['direction']} obstacle "
                        f"clearance={info['clearance']:.2f} m  threshold={info['threshold']:.2f} m  "
                        f"hard={info.get('hard_stop', -1.0):.2f} m  "
                        f"sector_pts={info['points']}  steer={steer:.2f}"
                    )
                    self._last_safety_msg_tick = self._tick
                return False

        self.last_safety_stop = None
        self.safety_stop_counter = 0
        apply_ctrl(self.vehicle, throttle=throttle, steer=steer, brake=brake, reverse=reverse)
        return True


#Final centering sequence:
#Stage 1: Pre-load steering while nearly stationary.
#Stage 2: Drive a short forward arc.
#Stage 3: Reverse straight a short distance to reduce lateral error.
    def _centering_control(self, pts, ex, ey, heading_err, spd):
        repeated_reverse_safety_limit = (
            self.last_safety_stop is not None and
            self.last_safety_stop.get("direction", "") == "reverse" and
            self.safety_stop_counter >= 3 and
            0.0 < self.last_safety_stop.get("clearance", -1.0) < 0.50
        )

        if abs(ey) > 2.0 and not repeated_reverse_safety_limit:
            steer_sign = 1.0 if ey > 0.0 else -1.0
            steer = clamp(0.88 * steer_sign + 0.018 * heading_err, -1.0, 1.0)
            thr = max(0.42, self._pid.step(spd, 2.5))
            self._safe_apply_ctrl(pts, throttle=thr, steer=steer, reverse=True)
            return

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\STAGE 1////////////////////////////////////
        if self._fwd_ctr_stage == "SET_STEER":
            steer = 0.9 if self._park_side == "right" else -0.9
            self._safe_apply_ctrl(pts, throttle=0.0, steer=steer, brake=0.25, reverse=False)
            if (self._tick - self._fwd_ctr_stage_tick) >= 8:
                self._fwd_ctr_stage = "FORWARD_ARC"
                self._fwd_ctr_stage_tick = self._tick
            return

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\STAGE 2////////////////////////////////////
        if self._fwd_ctr_stage == "FORWARD_ARC":
            base = 0.95 if self._park_side == "right" else -0.95
            steer = clamp(base + 0.04 * heading_err, -0.95, 0.95)
            thr = max(0.25, self._pid.step(spd, 1.8))
            self._safe_apply_ctrl(pts, throttle=thr, steer=steer, reverse=False)
            if abs(heading_err) < 4.0:
                self._fwd_ctr_stage = "REVERSE_STRAIGHT"
                self._fwd_ctr_stage_tick = self._tick
                self._ref = self.vehicle.get_location()
            return

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\STAGE 3////////////////////////////////////
        if self._fwd_ctr_stage == "REVERSE_STRAIGHT":
            reverse_dist = self._travel()
            reverse_done = (
                abs(ey) < CENTER_STAGE3_EY_DONE or
                reverse_dist >= CENTER_STAGE3_MAX_TRAVEL
            )
            if reverse_done:
                self._safe_apply_ctrl(pts, brake=1.0)
                return
            thr = max(0.18, self._pid.step(spd, 0.7))
            self._safe_apply_ctrl(pts, throttle=thr, steer=0.0, reverse=True)
            return
        

    def tick(self, pts: np.ndarray) -> bool:
        self._tick += 1
        spd = speed_kmh(self.vehicle)
        self._update_stall_counter(spd)

        if self._tick % DEBUG_EVERY_N == 0:
            dbg = self.detector.debug_info(pts)
            f_info = self._safety.clearance(pts, "forward", 0.0)
            r_info = self._safety.clearance(pts, "reverse", 0.0)
            dbg += (
                f"  safe(fwd={f_info['clearance']:.2f}m/{f_info['points']}pts, "
                f"rev={r_info['clearance']:.2f}m/{r_info['points']}pts)"
            )
            if self.slot and "target_world" in self.slot:
                ex, ey = self._target_error_local(self.slot["target_world"])
                he = self._heading_error(self.slot["target_yaw"])
                dbg += f"  err=(ex:{ex:.2f}, ey:{ey:.2f}, yaw:{he:.1f}°)"
            print(f"  [DBG t={self._tick:4d}] spd={spd:.1f}km/h  {dbg}")

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SCANNING////////////////////////////////////
        if self.state == self.SCANNING:
            thr = self._pid.step(spd, SCAN_SPEED_KMH)
            self._safe_apply_ctrl(pts, throttle=thr, steer=0.0, brake=0.0)

            result = self.detector.detect(pts)
            if result:
                self._slot_pose_from_detection(result) #to freeze the target world point
                gap = result["slot_end_x"] - result["slot_start_x"]
                self._log(
                    f"SLOT on {result['side'].upper()}  gap={gap:.1f}m  "
                    f"y={result['slot_y_dist']:.1f}m  start_x={result['slot_start_x']:.1f}m"
                )
                self._align_target = result["slot_start_x"] + VEHICLE_LENGTH * ALIGN_PASS_EXTRA
                self._go(self.FWD_ALIGN, f"Drive {self._align_target:.1f} m to align with slot …")

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\FORWARD ALIGN////////////////////////////////////
        elif self.state == self.FWD_ALIGN:
                    thr = self._pid.step(spd, SCAN_SPEED_KMH)
                    self._safe_apply_ctrl(pts, throttle=thr, steer=0.0, brake=0.0)

                    if self._travel() >= self._align_target:
                        self._go(self.STOP, "Full brake …")    
                                
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\STOP////////////////////////////////////
        elif self.state == self.STOP:
            self._safe_apply_ctrl(pts, brake=1.0)

            if spd < 0.5:
                self._park_side = self.slot.get("side", "right")
                print(f"  [DEBUG STOP] park_side='{self._park_side}'  slot_raw={self.slot}")

                first_dir = "RIGHT" if self._park_side == "right" else "LEFT"
                self._rev1_start = self.vehicle.get_location() 
                self._go(self.REV_RIGHT, f"Reversing phase-1 (steer {first_dir})")
       
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\REVERSE PHASE 1////////////////////////////////////
        elif self.state == self.REV_RIGHT:
            target = self.slot["target_world"]
            ex, ey = self._target_error_local(target)

            if self._park_side == "right":
                steer = 0.65   
            else:
                steer = -0.65  
 
            thr = self._pid.step(spd, MANEUVER_SPEED_KMH)
            moved = self._safe_apply_ctrl(pts, throttle=thr, steer=steer, reverse=True)

            rev1_travel = dist2d(self.vehicle.get_location(), self._rev1_start)
            yaw_now = self.vehicle.get_transform().rotation.yaw
            yaw_err_from_start = abs(normalize_angle_deg(yaw_now - self.slot["target_yaw"]))

            lateral_entered = (
                ey > REV_PHASE1_TRIGGER_EY
                if self._park_side == "right"
                else ey < -REV_PHASE1_TRIGGER_EY
               
            )

            yaw_reached = yaw_err_from_start > REV1_TARGET_YAW
            travel_limit_reached = rev1_travel > REV1_MAX_TRAVEL
            overshot_long = ex < REV_PHASE1_OVERSHOOT_EX
           
            print(
                f"  [DEBUG REVERSE_RIGHT] ex={ex:.2f}  ey={ey:.2f}  "
                f"steer={steer:.2f}  lateral_entered={lateral_entered}  park_side={self._park_side}"
                f"yaw_from_start={yaw_err_from_start:.1f} "
                f"lat={lateral_entered} yaw_hit={yaw_reached} travel_hit={travel_limit_reached}"
            )
            rear_stop, rear_info = self._safety.should_stop(
                pts, 
                reverse=True, 
                steer=steer, 
                speed_kmh_now=spd
                )

            if rear_stop and self._travel() > 0.8:
                second_dir = "LEFT" if self._park_side == "right" else "RIGHT"
                self._go(self.REV_LEFT, f"LiDAR stop in phase-1-> steer {second_dir}")
            elif lateral_entered or overshot_long or yaw_reached or travel_limit_reached:
                second_dir = "LEFT" if self._park_side == "right" else "RIGHT"
                self._go(self.REV_LEFT, f"Reversing phase-2 (steer {second_dir})")
            elif self._stall_ticks > STALL_TICKS_LIMIT:
                self._stall_ticks = 0
                self._safe_apply_ctrl(pts, brake=0.0, throttle=0.55, steer=steer, reverse=True)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\REVERSE PHASE 2////////////////////////////////////
        elif self.state == self.REV_LEFT:
            target = self.slot["target_world"]
            ex, ey = self._target_error_local(target)
            heading_err = self._heading_error(self.slot["target_yaw"])
            if self._park_side == "right":
                steer = -0.65  
            else:
                steer = 0.65   

            thr = self._pid.step(spd, MANEUVER_SPEED_KMH)
            moved = self._safe_apply_ctrl(pts, throttle=thr, steer=steer, reverse=True)

            print(
                f"  [DEBUG REVERSE_LEFT] ex={ex:.2f}  ey={ey:.2f}  "
                f"heading_err={heading_err:.1f}  steer={steer:.2f}  park_side={self._park_side}"
            )

            rear_stop, rear_info = self._safety.should_stop(
                pts, 
                reverse=True, 
                steer=steer, 
                speed_kmh_now=spd
                )
            
            near_parallel = abs(heading_err) < 5.0
            close_enough = abs(ey) < REVERSE_FINISH_EY and ex < REVERSE_FINISH_EX
           
            repeated_safety_limit = (
                self.last_safety_stop is not None and
                self.safety_stop_counter >= 3 and
                0.0 < self.last_safety_stop.get("clearance", -1.0) < 0.45
            )

            if rear_stop and self._travel() > 1.0:
                self._center_mode = None
                self._center_last_switch_tick = -10_000
                self._fwd_ctr_stage = "SET_STEER"
                self._fwd_ctr_stage_tick = self._tick
                self._go(self.FWD_CTR, "LiDAR stop reached in phase-2-> final centering")
            elif repeated_safety_limit :
                self._center_mode = None
                self._center_last_switch_tick = -10_000
                self._fwd_ctr_stage = "SET_STEER"
                self._fwd_ctr_stage_tick = self._tick
                self._go(self.FWD_CTR, "Safety-limited reverse accepted-> final centering")
            elif close_enough:
                self._center_mode = None
                self._center_last_switch_tick = -10_000
                self._fwd_ctr_stage = "SET_STEER"
                self._fwd_ctr_stage_tick = self._tick
                self._go(self.FWD_CTR, "Final centering...")
            elif self._stall_ticks > STALL_TICKS_LIMIT:
                self._stall_ticks = 0
                self._safe_apply_ctrl(pts, brake=0.0, throttle=0.55, steer=steer, reverse=True)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\FORWARD CENTER////////////////////////////////////
        elif self.state == self.FWD_CTR:
            target = self.slot["target_world"]
            ex, ey = self._target_error_local(target)
            heading_err = self._heading_error(self.slot["target_yaw"])

            self._centering_control(pts, ex, ey, heading_err, spd)

            done_pose = (
                abs(ex) < TARGET_EX_TOL and
                abs(ey) < TARGET_EY_TOL and
                abs(heading_err) < TARGET_YAW_TOL
            )

            timeout_pose = (
                self._travel() >= DIST_FWD_CTR and 
                abs(ey) < 0.35 and #0.35
                abs(heading_err) < 8.0  and #8
                abs(ex) < 1.0 
            )


            safety_limited_done = (
                self.last_safety_stop is not None and
                self.safety_stop_counter >= 3 and
                0.0 < self.last_safety_stop.get("clearance", -1.0) < 0.45 and
                abs(ey) < max(0.65, TARGET_EY_TOL + 0.15) and
                abs(heading_err) < max(10.0, TARGET_YAW_TOL + 4.0)
            )

            if done_pose or timeout_pose or safety_limited_done:
                self._safe_apply_ctrl(pts, brake=1.0)
                if done_pose:
                    note = "Parking complete!"
                elif safety_limited_done:
                    note = "Parking complete (safety-limited final pose)."
                else:
                    note = "Parking finished (best pose reached)."
                self._go(self.PARKED, note)

            elif self._stall_ticks > STALL_TICKS_LIMIT:
                self._stall_ticks = 0

                good_enough_after_stall = (
                    abs(ex) < 1.4 and
                    abs(ey) < 1.45 and
                    abs(heading_err) < 14.0
                )

                if good_enough_after_stall:
                    self._safe_apply_ctrl(pts, brake=1.0)
                    self._go(self.PARKED, "Parking finished after stalled centering (good enough pose).")
                else:
                    self._safe_apply_ctrl(pts, brake=1.0)
                    self._go(self.PARKED, "Parking stopped at best feasible pose.")
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\PARKED////////////////////////////////////
        elif self.state == self.PARKED:
            self._safe_apply_ctrl(pts, brake=1.0)
            return True

        return False



#*********************************BEV VISUALISER FUNCTION*********************************
#*****************************************************************************************
def draw_bev(pts, slot, state, size=WIN_W):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    S   = size / 30.0
    cx  = size // 2
    cy  = int(size * 0.6)

    for p in pts:
        px = int(cx + p[1] * S)
        py = int(cy - p[0] * S)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (50, 200, 80)

    cv2.rectangle(img, (cx-8, cy-16), (cx+8, cy+16), (30, 120, 255), 2)
    cv2.putText(img, "EGO", (cx+10, cy+6), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (30, 120, 255), 1)

    if slot:
        sx   = slot["slot_start_x"];  ex = slot["slot_end_x"]
        yd   = slot["slot_y_dist"]
        # In BEV: px = cx + world_Y * S  → right side = positive Y → sign = +1
        sign = 1 if slot.get("side", "right") == "right" else -1
        lx   = int(cx + sign * yd * S)
        y1   = int(cy - ex * S);  y2 = int(cy - sx * S)
        cv2.rectangle(img, (lx-10, y1), (lx+10, y2), (0, 60, 255), 2)
        cv2.putText(img, "SLOT", (lx-38, (y1+y2)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    cv2.putText(img, "FWD",  (cx+4, 14),  cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1)
    cv2.putText(img, state,  (4, size-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)
    return img


#*********************************SPAWN BLOCKER CARS FUNCTION*********************************
#**********************************************************************************************
def spawn_parked_cars(world, ego_tf):
    bp_lib = world.get_blueprint_library()
    actors = []

    car_configs = [
        (PARK_CAR_A_X, PARK_CAR_A_Y, PARK_CAR_A_Z, PARK_CAR_A_YAW, PARK_CAR_A_BP, PARK_CAR_A_COLOR),
        (PARK_CAR_B_X, PARK_CAR_B_Y, PARK_CAR_B_Z, PARK_CAR_B_YAW, PARK_CAR_B_BP, PARK_CAR_B_COLOR),
    ]

    for x, y, z, yaw, bp_id, color in car_configs:
        bp = bp_lib.find(bp_id)
        if bp is None:
            print(f"  WARNING! blueprint '{bp_id}' not found, skipping.")
            continue
        if bp.has_attribute("color"):
            bp.set_attribute("color", color)

        tf = carla.Transform(carla.Location(x=x, y=y, z=z),
                             carla.Rotation(yaw=yaw))
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            tf.location.z += 0.3
            actor = world.try_spawn_actor(bp, tf)
        if actor:
            actor.set_simulate_physics(False)
            actors.append(actor)
            print(f"  Blocker: {bp_id}  ({x}, {y})")
        else:
            print(f"  WARNING! spawn failed for {bp_id} at ({x}, {y})")

    return actors



#*********************************RESET EGO VEHICLE FUNCTION*********************************
#**********************************************************************************************
def reset_ego(vehicle, fsm, detector):
    spawn_tf = carla.Transform(
        carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
        carla.Rotation(yaw=SPAWN_YAW))
    
    vehicle.set_transform(spawn_tf)
    apply_ctrl(vehicle, brake=1.0)


    fsm.state   = fsm.SCANNING
    fsm.slot    = None
    fsm._park_side = "right"
    fsm._ref    = vehicle.get_location()
    fsm._tick   = 0
    fsm._stall_ticks = 0
    fsm.last_safety_stop = None
    fsm.safety_stop_counter = 0
    fsm._center_mode = None
    fsm._center_last_switch_tick = -10_000
    detector.side = None   

    print("\n🔄  RESET: ego teleported back to start. Press R again to re-reset.\n")


#**********************************************************************
#*********************************MAIN*********************************
#**********************************************************************
def main():
    world   = None
    vehicle = None
    sensors = []
    parked  = []
 
    try:
        print(f"\nConnecting to CARLA  {HOST}:{PORT}... ")
        client = carla.Client(HOST, PORT)
        client.set_timeout(30.0)
        print(f"Server: {client.get_server_version()}")
 
        # Load layered variant
        print(f"\nLoading {TOWN}_Opt...")
        try:
            world = client.load_world(f"{TOWN}_Opt")
            print(f"  Loaded {TOWN}_Opt")
        except Exception:
            print(f"  Falling back to {TOWN}")
            world = client.load_world(TOWN)

        time.sleep(3.0)
 
        settings                     = world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        print("Sync mode ON (20 fps)")

        strip_map(world)
        world.tick()
        time.sleep(1.0)
 
        bp_lib = world.get_blueprint_library()
 
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SPAWN EGO VEHICLE////////////////////////////////////
        ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
        ego_bp.set_attribute("color", "200,200,200")
 
        spawn_tf = carla.Transform(
            carla.Location(x=SPAWN_X, y=SPAWN_Y, z=SPAWN_Z),
            carla.Rotation(yaw=SPAWN_YAW))
 
        vehicle = world.try_spawn_actor(ego_bp, spawn_tf)
        
        if vehicle is None:
            spts = sorted(
                world.get_map().get_spawn_points(),
                key=lambda s: dist2d(s.location,
                carla.Location(x=SPAWN_X, y=SPAWN_Y))
                )
            vehicle = world.spawn_actor(ego_bp, spts[0])
            spawn_tf = spts[0]
            print(f"  Ego at fallback: {spts[0].location}")
        else:
            print(f"  Ego at ({SPAWN_X}, {SPAWN_Y})")
 
        vehicle.set_autopilot(False)
        apply_ctrl(vehicle, brake=0.0)
        world.tick()

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SPAWN BLOCKER VEHICLES//////////////////////////////////// 
        print("\nSpawning blocker cars...")
        parked = spawn_parked_cars(world, vehicle.get_transform())
        print(f"  {len(parked)}/2 spawned")
        for _ in range(5): 
            world.tick()
 
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\SENSORS////////////////////////////////////
        bridge   = SensorBridge()
 
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range",              str(LIDAR_RANGE))
        lidar_bp.set_attribute("channels",           str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("points_per_second",  str(LIDAR_PPS))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_FPS))
        lidar_bp.set_attribute("upper_fov",          "15.0")
        lidar_bp.set_attribute("lower_fov",         "-25.0")
        lidar = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=vehicle
            )
        lidar.listen(bridge.on_lidar)
        sensors.append(lidar)
 
        seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(WIN_W))
        seg_bp.set_attribute("image_size_y", str(WIN_H))
        seg_bp.set_attribute("fov", "90")

        seg_cam = world.spawn_actor(
            seg_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=vehicle
            )
        seg_cam.listen(bridge.on_seg)
        sensors.append(seg_cam)
 
        print("\nWaiting for sensor warm-up...", end="", flush=True)
        for _ in range(80):
            world.tick()
            _, cnt = bridge.lidar()
            if cnt >= 3: break
            print(".", end="", flush=True)
        print(" OK")
 
        pts0, _ = bridge.lidar()
        d_pre   = SlotDetector()
        print(f"  {d_pre.debug_info(pts0)}\n")

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\WINDOWS//////////////////////////////////// 
        cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("LiDAR Bird-Eye View",   cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Semantic Segmentation", WIN_W, WIN_H)
        cv2.resizeWindow("LiDAR Bird-Eye View",   WIN_W, WIN_W)
 
        print("-" * 60)
        print("Autonomous parallel parking started.  ESC = abort.\n")
        print(f"  DIST_FWD_CTR={DIST_FWD_CTR}m  EX_TOL={TARGET_EX_TOL}m  EY_TOL={TARGET_EY_TOL}m  YAW_TOL={TARGET_YAW_TOL}°\n")
 
        detector = SlotDetector(min_length=MIN_SLOT_LENGTH)
        fsm      = ParkingFSM(vehicle, detector)
        done     = False
 
        while not done:
            world.tick()
            pts, _ = bridge.lidar()
            seg    = bridge.seg()
 
            done = fsm.tick(pts)
 
            disp = seg.copy()
            cv2.putText(
                disp,
                f"State: {fsm.state}   {speed_kmh(vehicle):.1f} km/h",
                (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.65, 
                (255,255,255), 
                2
                )
            cv2.putText(
                disp,
                "R = Reset   ESC = Quit",
                (10, WIN_H - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 220, 220), 
                1
                )
            
            cv2.imshow("Semantic Segmentation", disp)
            cv2.imshow("LiDAR Bird-Eye View", draw_bev(pts, fsm.slot, fsm.state))
 
            key = cv2.waitKey(1)
            if key == 27:                              
                print("\nESC abort.")
                break
            elif key == ord('r') or key == ord('R'):   
                done = False
                reset_ego(vehicle, fsm, detector)
 
        if done:
            print("\nParking complete! Holding...")
            print("   Press R to reset and try again, ESC to quit.")

            hold_ticks = int(FINAL_HOLD_SECONDS / 0.05)

            for _ in range(hold_ticks):
                apply_ctrl(vehicle, brake=1.0)
                world.tick()
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == ord('r') or key == ord('R'):
                    done = False
                    reset_ego(vehicle, fsm, detector)
                    break
 
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
    finally:
        print("\nCleaning up ...")
        cv2.destroyAllWindows()

        if world:
            try:
                for layer in UNLOAD_LAYERS:
                    try: 
                        world.load_map_layer(layer)
                    except Exception: 
                        pass
                    
                s = world.get_settings()
                s.synchronous_mode = False
                world.apply_settings(s)
            except Exception: 
                pass
        for s in sensors:
            try: 
                s.stop() 
                s.destroy()
            except Exception: 
                pass
        for p in parked:
            try: 
                p.destroy()
            except Exception: 
                pass
        if vehicle:
            try: 
                vehicle.destroy()
            except Exception: 
                pass
        print("Done.")
 
 
if __name__ == "__main__":
    main()
 