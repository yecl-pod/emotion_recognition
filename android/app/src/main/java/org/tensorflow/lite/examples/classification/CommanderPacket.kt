package org.tensorflow.lite.examples.classification

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
