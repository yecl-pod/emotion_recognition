package org.tensorflow.lite.examples.classification

import android.util.Log
import java.nio.ByteBuffer

class CommanderPacket(@JvmField val roll: Float, @JvmField val pitch: Float, @JvmField val yaw: Float, @JvmField val thrust: Float): CrtpPacket(0, CrtpPort.COMMANDER) {
    var mRoll: Float = roll // +left/-right
    var mPitch: Float = pitch // +forward/-backward
    var mYaw: Float = yaw // +rotate clockwise/-counter clockwise
    var mThrust: UShort = thrust.toInt().toUShort() // multiplier
    
    // make sure (mRoll + mPitch) * mThrust < 14000

    override fun getDataByteCount(): Int {
        return (3 * 4 + 1 * 2)
    }

    override fun serializeData(b: ByteBuffer) {
        b.putFloat(mRoll)
        b.putFloat(mPitch)
        b.putFloat(mYaw)
        b.putShort(mThrust.toShort())
    }

    override fun toString(): String {
        return "CommanderPacket - R: $mRoll, P: $mPitch, Y: $mYaw, T: $mThrust"
    }
}

class CommanderHoverPacket(vx: Float, vy: Float, vyaw: Float, dis: Float): CrtpPacket(1, CrtpPort.COMMANDER_GENERIC) {
    private var mVx: Float = vx
    private var mVy: Float = vy
    private var mVyaw: Float = vyaw
    private var mDis: Float = dis
    private val mTYPE: Byte = 5 // type of hover command

    override fun getDataByteCount(): Int? {
        return (4 * 4 + 1 * 1) // 4 float, 1 unsigned char
    }

    override fun serializeData(b: ByteBuffer) {
        b.put(mTYPE)
        b.putFloat(mVx)
        b.putFloat(mVy)
        b.putFloat(mVyaw)
        b.putFloat(mDis)
    }

    override fun toString(): String {
        return "CommanderHoverPacket - Vx: $mVx, Vy: $mVy, Vyaw: $mVyaw, Dis: $mDis"
    }
}